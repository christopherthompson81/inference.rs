//! 3D mrope position ids, per transformers `PaddleOCRVLModel.get_rope_index` (Qwen2-VL style, images only).

use inference_tensor::{Device, Result, Tensor};

// Grids are pre-merge, one per run of t*(h/m)*(w/m) placeholders; decode adds the returned delta to its position.
fn prompt_positions(
    input_ids: &[u32],
    image_grids: &[(usize, usize, usize)],
    image_token_id: u32,
    merge: usize,
) -> ([Vec<i64>; 3], i64) {
    let seq = input_ids.len();
    let mut rows: [Vec<i64>; 3] = [
        Vec::with_capacity(seq),
        Vec::with_capacity(seq),
        Vec::with_capacity(seq),
    ];
    let mut next: i64 = 0; // max emitted position + 1
    let mut st = 0usize;

    let emit_text = |rows: &mut [Vec<i64>; 3], start: i64, len: usize| {
        for k in 0..len as i64 {
            for row in rows.iter_mut() {
                row.push(start + k);
            }
        }
    };

    for &(t, h, w) in image_grids {
        let ed = (st..seq)
            .find(|&i| input_ids[i] == image_token_id)
            .unwrap_or(seq);
        let text_len = ed - st;
        emit_text(&mut rows, next, text_len);
        next += text_len as i64;

        let (llm_t, llm_h, llm_w) = (t, h / merge, w / merge);
        let base = next;
        for ti in 0..llm_t as i64 {
            for hi in 0..llm_h as i64 {
                for wi in 0..llm_w as i64 {
                    rows[0].push(base + ti);
                    rows[1].push(base + hi);
                    rows[2].push(base + wi);
                }
            }
        }
        let span = llm_t.max(llm_h).max(llm_w) as i64;
        next = base + span;
        st = ed + llm_t * llm_h * llm_w;
    }

    if st < seq {
        emit_text(&mut rows, next, seq - st);
    }

    debug_assert_eq!(
        rows[0].len(),
        seq,
        "position_ids length must equal sequence length"
    );
    let max_pos = rows.iter().flatten().copied().max().unwrap_or(-1);
    (rows, max_pos + 1 - seq as i64)
}

/// How far past its index a token after the prompt sits, as decode positions it (index + delta).
pub fn position_delta(
    prompt: &[u32],
    grids: &[(usize, usize, usize)],
    image_token_id: u32,
    merge: usize,
) -> i64 {
    prompt_positions(prompt, grids, image_token_id, merge).1
}

/// The `[3, batch, len]` positions of each row's `len` tokens from its offset: the prompt's own positions inside the
/// prompt, and past it the cursor carried on (index + delta), as transformers' `get_rope_index` gives them. Built on
/// the host from each row's prompt, so a step uploads only its own window.
pub fn window_positions(
    prompts: &[Vec<u32>],
    grids: &[Vec<(usize, usize, usize)>],
    image_token_id: u32,
    merge: usize,
    offsets: &[usize],
    len: usize,
    dev: &Device,
) -> Result<Tensor> {
    let batch = prompts.len();
    let mut planes: [Vec<i64>; 3] = std::array::from_fn(|_| Vec::with_capacity(batch * len));
    for (b, prompt) in prompts.iter().enumerate() {
        let (rows, delta) = prompt_positions(prompt, &grids[b], image_token_id, merge);
        for (plane, row) in planes.iter_mut().zip(&rows) {
            plane.extend((offsets[b]..offsets[b] + len).map(|i| match row.get(i) {
                Some(&pos) => pos,
                None => i as i64 + delta,
            }));
        }
    }
    Tensor::from_vec(planes.concat(), (3, batch, len), dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: u32 = 999;
    const GRID: (usize, usize, usize) = (1, 4, 4);

    fn at(positions: &Tensor, row: usize, col: usize) -> Vec<i64> {
        positions
            .narrow(1, row, 1)
            .and_then(|t| t.narrow(2, col, 1))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<i64>())
            .unwrap()
    }

    // Needs a non-zero delta (image compresses positions) to not be vacuous.
    #[test]
    fn decode_position_continues_prefill_cursor() {
        let prompt = vec![10, 11, IMG, IMG, IMG, IMG, 12, 13];
        let (_, delta) = prompt_positions(&prompt, &[GRID], IMG, 2);
        assert!(
            delta < 0,
            "image block should compress positions -> negative delta"
        );
        let mut extended = prompt.clone();
        extended.push(42);
        let (rows, _) = prompt_positions(&extended, &[GRID], IMG, 2);
        let decoded = window_positions(
            std::slice::from_ref(&prompt),
            &[vec![GRID]],
            IMG,
            2,
            &[prompt.len()],
            1,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(
            at(&decoded, 0, 0),
            vec![rows[0][prompt.len()]; 3],
            "decode must continue the cursor"
        );
    }

    // graph decode reads index + position_delta from the step's positions instead of building the planes
    #[test]
    fn position_delta_gives_the_decode_planes() {
        let prompt = vec![10, 11, IMG, IMG, IMG, IMG, 12, 13];
        let delta = position_delta(&prompt, &[GRID], IMG, 2);
        for offset in [prompt.len(), prompt.len() + 5] {
            let decoded = window_positions(
                std::slice::from_ref(&prompt),
                &[vec![GRID]],
                IMG,
                2,
                &[offset],
                1,
                &Device::Cpu,
            )
            .unwrap();
            assert_eq!(at(&decoded, 0, 0), vec![offset as i64 + delta; 3]);
        }
    }

    #[test]
    fn a_window_takes_its_columns_of_each_prompt() {
        let row0 = vec![10, 11, IMG, IMG, IMG, IMG, 12, 13];
        let row1 = vec![20, 21, 22, 23, 24, 25, 26, 27];
        let grids = vec![vec![GRID], vec![]];
        let (full0, _) = prompt_positions(&row0, &grids[0], IMG, 2);
        let (full1, _) = prompt_positions(&row1, &grids[1], IMG, 2);
        let window =
            window_positions(&[row0, row1], &grids, IMG, 2, &[2, 2], 4, &Device::Cpu).unwrap();
        for col in 0..4 {
            assert_eq!(
                at(&window, 0, col),
                (0..3).map(|p| full0[p][2 + col]).collect::<Vec<_>>()
            );
            assert_eq!(
                at(&window, 1, col),
                (0..3).map(|p| full1[p][2 + col]).collect::<Vec<_>>()
            );
        }
    }
}
