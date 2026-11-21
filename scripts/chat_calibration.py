#!/usr/bin/env python3
"""Chat-formatted calibration and eval text from a model's own replies (#1013 item 4).

`prompts` draws first user turns from UltraChat 200k's test split and MBPP tasks (parquet files downloaded beforehand),
shuffled with a fixed seed, into a calibration set and a held-out eval set. `generate` asks an OpenAI-compatible
server (`inference serve` with the model) for its replies, thinking on for 3 prompts in 4, resuming where an earlier
run stopped. `render` writes the conversations through the model's own chat template (read from its GGUF), a reply
cut at the token limit kept as generated; `--tokens` cuts the text at a conversation boundary to that many tokens.

The calibration investigation (Run 2) found half of the WikiText-plus-docs mix and half of this text the best of the
mixes tried: 16% lower KLD on chat, 4% higher on WikiText.

Usage:
  scripts/chat_calibration.py prompts --ultrachat DIR --mbpp DIR --out DIR [--cal 700 --cal-code 100]
      [--eval 160 --eval-code 40]
  scripts/chat_calibration.py generate --url http://localhost:1234 --prompts prompts-cal.json --out gen-cal.jsonl
  scripts/chat_calibration.py render --gguf model.gguf --gen gen-cal.jsonl --out chat-cal.txt
      [--tokens N --tokenizer tokenizer.json]
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import re
import sys
import urllib.request

SEED = 0
MIN_PROMPT_CHARS = 20
MAX_PROMPT_CHARS = 3000
MAX_TOKENS = 2048
TEMPERATURE = 0.6
TOP_P = 0.95
THINK_EVERY = 4  # thinking off for one prompt in this many
CONCURRENCY = 16
REQUEST_TIMEOUT_S = 1800
USER_TURN = '<|im_start|>user'


def prompts(args):
    import pyarrow.parquet as pq
    rows = []
    for f in glob.glob(f'{args.ultrachat}/**/test_sft*.parquet', recursive=True):
        rows += pq.read_table(f, columns=['prompt']).column('prompt').to_pylist()
    rows = [p.strip() for p in rows if MIN_PROMPT_CHARS < len(p) < MAX_PROMPT_CHARS]
    tasks = []
    for f in glob.glob(f'{args.mbpp}/**/*.parquet', recursive=True):
        t = pq.read_table(f)
        tasks += [f'{text}\n\nWrite a Python function for this. Tests:\n' + '\n'.join(tests)
                  for text, tests in zip(t.column('text').to_pylist(), t.column('test_list').to_pylist())]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    rng.shuffle(tasks)
    cal = rows[:args.cal] + tasks[:args.cal_code]
    held = rows[args.cal:args.cal + args.eval] + tasks[args.cal_code:args.cal_code + args.eval_code]
    for name, v in (('cal', cal), ('eval', held)):
        rng.shuffle(v)
        with open(os.path.join(args.out, f'prompts-{name}.json'), 'w') as f:
            json.dump(v, f)
        print(name, len(v))


def generate(args):
    with open(args.prompts) as f:
        ps = json.load(f)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)['i'] for line in f}

    def ask(i):
        think = i % THINK_EVERY != THINK_EVERY - 1
        body = {'model': 'default', 'messages': [{'role': 'user', 'content': ps[i]}], 'max_tokens': MAX_TOKENS,
                'temperature': TEMPERATURE, 'top_p': TOP_P, 'enable_thinking': think}
        req = urllib.request.Request(args.url + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        choice = json.load(urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S))['choices'][0]
        return {'i': i, 'think': think, 'prompt': ps[i], 'reasoning': choice['message'].get('reasoning_content'),
                'content': choice['message'].get('content'), 'finish': choice.get('finish_reason')}

    todo = [i for i in range(len(ps)) if i not in done]
    with open(args.out, 'a') as f, cf.ThreadPoolExecutor(args.concurrency) as ex:
        for n, r in enumerate(ex.map(ask, todo)):
            f.write(json.dumps(r) + '\n')
            f.flush()
            print(f'{n + 1}/{len(todo)}', end='\r', flush=True)
    print()


def render(args):
    sys.path.insert(0, os.path.expanduser('~/Programming/llama.cpp/gguf-py'))
    import gguf
    import jinja2
    template = gguf.GGUFReader(args.gguf).fields['tokenizer.chat_template'].contents()
    # HF's assistant-mask tags are not jinja
    template = re.sub(r'{%-?\s*(end)?generation\s*-?%}', '', template)
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
    env.filters['tojson'] = lambda v, **_: json.dumps(v, ensure_ascii=False)

    def raise_exception(message):
        raise ValueError(message)

    tpl = env.from_string(template)
    convs = []
    with open(args.gen) as f:
        for line in f:
            g = json.loads(line)
            user = [{'role': 'user', 'content': g['prompt']}]
            if g['finish'] != 'stop':
                # cut at the token limit: the prompt, then the raw generation as far as it got
                text = tpl.render(messages=user, add_generation_prompt=True, enable_thinking=g['think'],
                                  raise_exception=raise_exception)
                convs.append(text + (g['reasoning'] or '') + (g['content'] or '') + '\n')
                continue
            reply = {'role': 'assistant', 'content': g['content']}
            if g['reasoning']:
                reply['reasoning_content'] = g['reasoning']
            convs.append(tpl.render(messages=user + [reply], add_generation_prompt=False, enable_thinking=g['think'],
                                    raise_exception=raise_exception))
    if args.tokens:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
        kept, total = [], 0
        for c in convs:
            n = len(tok.encode(c, add_special_tokens=False).ids)
            if total + n > args.tokens:
                break
            kept.append(c)
            total += n
        convs = kept
        print(f'{len(convs)} conversations, {total} tokens')
    with open(args.out, 'w') as f:
        f.write(''.join(convs))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prompts')
    p.add_argument('--ultrachat', required=True)
    p.add_argument('--mbpp', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cal', type=int, default=700, help='chat prompts for calibration')
    p.add_argument('--cal-code', type=int, default=100, help='MBPP tasks for calibration')
    p.add_argument('--eval', type=int, default=160, help='chat prompts held out')
    p.add_argument('--eval-code', type=int, default=40, help='MBPP tasks held out')
    g = sub.add_parser('generate')
    g.add_argument('--url', required=True)
    g.add_argument('--prompts', required=True)
    g.add_argument('--out', required=True)
    g.add_argument('--concurrency', type=int, default=CONCURRENCY)
    r = sub.add_parser('render')
    r.add_argument('--gguf', required=True)
    r.add_argument('--gen', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--tokens', type=int)
    r.add_argument('--tokenizer')
    args = a.parse_args()
    {'prompts': prompts, 'generate': generate, 'render': render}[args.cmd](args)


if __name__ == '__main__':
    main()
