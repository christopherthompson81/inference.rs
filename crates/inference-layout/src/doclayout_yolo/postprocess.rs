use crate::pp_doclayout_v3::LayoutDetection;

use super::preprocess::Letterbox;

/// Detections the one-to-one head keeps per image before the score threshold.
pub const MAX_DETECTIONS: usize = 300;

/// The v10 head's selection without NMS: the `MAX_DETECTIONS` anchors with the best class score, then the best
/// `MAX_DETECTIONS` (anchor, class) pairs among them, so an anchor can come back under two classes.
/// `boxes` is `[x1, y1, x2, y2]` per anchor in input pixels, `scores` is `(classes, anchors)` row-major.
pub fn select(
    boxes: &[[f32; 4]],
    scores: &[f32],
    classes: usize,
    threshold: f32,
) -> Vec<(usize, usize, f32)> {
    let anchors = boxes.len();
    let best = |a: usize| {
        (0..classes)
            .map(|c| scores[c * anchors + a])
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let mut order: Vec<usize> = (0..anchors).collect();
    order.sort_by(|&a, &b| best(b).total_cmp(&best(a)));
    order.truncate(MAX_DETECTIONS);
    let mut pairs: Vec<(usize, usize, f32)> = order
        .iter()
        .flat_map(|&a| (0..classes).map(move |c| (a, c, scores[c * anchors + a])))
        .collect();
    pairs.sort_by(|x, y| y.2.total_cmp(&x.2));
    pairs.truncate(MAX_DETECTIONS);
    pairs.retain(|p| p.2 > threshold);
    pairs
}

/// Detections in source pixels, ranked by score, with a reading order from `reading_order`.
pub fn detections(
    boxes: &[[f32; 4]],
    scores: &[f32],
    labels: &[String],
    threshold: f32,
    letterbox: &Letterbox,
) -> Vec<LayoutDetection> {
    let mut dets: Vec<LayoutDetection> = select(boxes, scores, labels.len(), threshold)
        .into_iter()
        .map(|(anchor, class_id, score)| {
            let bbox = letterbox.to_source(boxes[anchor]);
            LayoutDetection {
                class_id,
                label: labels[class_id].clone(),
                score,
                bbox,
                reading_order: 0,
                polygon: corners(bbox),
                query: anchor,
            }
        })
        .collect();
    let bboxes: Vec<[f32; 4]> = dets.iter().map(|d| d.bbox).collect();
    for (rank, i) in reading_order(&bboxes).into_iter().enumerate() {
        dets[i].reading_order = rank;
    }
    dets.sort_by_key(|d| d.reading_order);
    dets
}

fn corners([x1, y1, x2, y2]: [f32; 4]) -> Vec<[f32; 2]> {
    vec![[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
}

/// Recursive XY-cut: split into horizontal bands, else into columns, top to bottom and left to right; regions no cut
/// separates read top to bottom. Returns box indices in reading order.
pub fn reading_order(boxes: &[[f32; 4]]) -> Vec<usize> {
    let mut out = Vec::with_capacity(boxes.len());
    xy_cut(boxes, (0..boxes.len()).collect(), &mut out);
    out
}

fn xy_cut(boxes: &[[f32; 4]], ids: Vec<usize>, out: &mut Vec<usize>) {
    if ids.len() <= 1 {
        out.extend(ids);
        return;
    }
    let bands = bands(boxes, &ids);
    let groups = if bands.len() > 1 {
        bands
    } else {
        split(boxes, &ids, 0)
    };
    if groups.len() > 1 {
        for group in groups {
            xy_cut(boxes, group, out);
        }
        return;
    }
    let mut ids = ids;
    ids.sort_by(|&a, &b| {
        boxes[a][1]
            .total_cmp(&boxes[b][1])
            .then(boxes[a][0].total_cmp(&boxes[b][0]))
    });
    out.extend(ids);
}

/// Horizontal bands, with neighbouring multi-column bands that share a gutter kept together: the rows of two aligned
/// columns are one band, read column by column, not row by row. A single-column band (a heading, a footer line)
/// separates.
fn bands(boxes: &[[f32; 4]], ids: &[usize]) -> Vec<Vec<usize>> {
    let columns = |ids: &[usize]| split(boxes, ids, 0).len() > 1;
    let mut bands: Vec<Vec<usize>> = Vec::new();
    for band in split(boxes, ids, 1) {
        if let Some(last) = bands.last_mut()
            && columns(last)
            && columns(&band)
        {
            let joined: Vec<usize> = last.iter().chain(&band).copied().collect();
            if columns(&joined) {
                *last = joined;
                continue;
            }
        }
        bands.push(band);
    }
    bands
}

/// The runs of `ids` whose extents along `axis` (0 = x, 1 = y) overlap, in order along it.
fn split(boxes: &[[f32; 4]], ids: &[usize], axis: usize) -> Vec<Vec<usize>> {
    let mut sorted = ids.to_vec();
    sorted.sort_by(|&a, &b| boxes[a][axis].total_cmp(&boxes[b][axis]));
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut end = f32::NEG_INFINITY;
    for id in sorted {
        let (lo, hi) = (boxes[id][axis], boxes[id][axis + 2]);
        match groups.last_mut() {
            Some(group) if lo < end => group.push(id),
            _ => groups.push(vec![id]),
        }
        end = end.max(hi);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_columns_under_a_title_read_down_each_column() {
        let boxes = [
            [520., 300., 1000., 600.], // right column, top
            [0., 0., 1000., 100.],     // title
            [0., 650., 480., 900.],    // left column, bottom
            [0., 300., 480., 600.],    // left column, top
            [520., 640., 1000., 950.], // right column, bottom
            [0., 1000., 1000., 1050.], // footer
        ];
        assert_eq!(reading_order(&boxes), [1, 3, 2, 0, 4, 5]);
    }

    #[test]
    fn a_margin_page_number_stays_below_the_footer_line_above_it() {
        let boxes = [
            [340., 2117., 1450., 2173.], // footnote across the text block
            [107., 2260., 136., 2280.],  // page number in the margin
            [361., 2257., 909., 2283.],  // running footer beside it
        ];
        assert_eq!(reading_order(&boxes), [0, 1, 2]);
    }

    #[test]
    fn selection_keeps_both_classes_of_an_anchor_and_drops_low_scores() {
        let boxes = [[0., 0., 1., 1.], [2., 2., 3., 3.], [4., 4., 5., 5.]];
        // (classes, anchors): anchor 0 scores 0.9 as class 0 and 0.6 as class 1
        let scores = [0.9, 0.1, 0.3, 0.6, 0.2, 0.05];
        assert_eq!(
            select(&boxes, &scores, 2, 0.25),
            [(0, 0, 0.9), (0, 1, 0.6), (2, 0, 0.3)]
        );
    }
}
