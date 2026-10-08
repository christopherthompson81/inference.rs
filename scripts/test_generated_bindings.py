"""The bindings' native declarations are current with inference.h, and the generator's type rules hold."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings" / "scripts"))
import generate_native

VERSION = """
#define INFERENCE_ABI_VERSION_MAJOR 0
#define INFERENCE_ABI_VERSION_MINOR 0
#define INFERENCE_ABI_VERSION_PATCH 1
typedef enum inference_status { INFERENCE_OK = 0 } inference_status;
typedef struct inference_engine inference_engine;
typedef struct inference_string inference_string;
"""


def csharp(declarations: str) -> str:
    return generate_native.csharp_methods(generate_native.Header(VERSION + declarations))


class Generated(unittest.TestCase):
    def test_the_committed_declarations_are_current(self):
        self.assertEqual(generate_native.stale(), [], "run bindings/scripts/generate_native.py")

    def test_a_declaration_broken_across_lines_is_parsed(self):
        rendered = csharp("INFERENCE_API\ninference_status inference_probe (const inference_engine *engine);\n")
        self.assertIn("inference_probe(IntPtr engine)", rendered)

    def test_a_counted_output_is_an_array_and_a_single_one_an_out(self):
        rendered = csharp(
            "INFERENCE_API inference_status inference_probe(uint32_t *out_tokens, size_t capacity,"
            " int32_t *out_count);\n"
        )
        self.assertIn("uint* outTokens, nuint capacity, out int outCount", rendered)

    def test_an_input_array_of_strings_is_not_an_out(self):
        rendered = csharp("INFERENCE_API inference_status inference_probe(const char **names, size_t count);\n")
        self.assertIn("IntPtr* names, nuint count", rendered)

    def test_only_a_length_makes_a_buffer(self):
        rendered = csharp(
            "INFERENCE_API inference_status inference_probe(const char *name, size_t index,"
            " const char *data, size_t data_len);\n"
        )
        self.assertIn("string? name, nuint index, byte* data, nuint dataLen", rendered)

    def test_a_declaration_the_parser_cannot_read_is_an_error(self):
        with self.assertRaises(ValueError):
            generate_native.Header(VERSION + "INFERENCE_API int (*inference_probe)(void);\n")


if __name__ == "__main__":
    unittest.main()
