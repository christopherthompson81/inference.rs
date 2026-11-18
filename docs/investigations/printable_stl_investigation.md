# Printable STL from the image jobs

Goal: a closed, single-shell solid (STL, millimetres) from both image jobs, Idol's Gaussian splats and TRELLIS.2's
result, to 3D print. Code: `crates/inference-recon/src/printable.rs`, `crates/inference-jobs/src/print.rs`.

These entries were written up after the runs, from the session's commands and outputs; the times are approximate.

## Run 1 - 2026-11-18 15:00

Question: does Idol's octree give a dense occupancy to mesh directly?

No: `OctreeDecoder::sample` splits a budget of anchor points (8,192 by default) among children level by level; the
level-8 leaves are a sparse sample of the surface, not an occupancy. Dropped; the Gaussians themselves are the source.

## Run 2 - 2026-11-18 15:20

Question: does summed Gaussian density, sealed by a flood from outside and meshed by surface nets, give a closed solid?

`generates_gaussians_end_to_end` (armchair image, 262,144 Gaussians), 256^3 grid, density threshold 0.5, each
Gaussian's thinnest axis floored at 0.6 voxel (thinner sheets slip between grid points), blur sigma 0.8:

    printable: volume 0.0277, 114276 faces, solid 0.79 s, surface 0.41 s
    trimesh: watertight False, bodies 2; edge use counts {2: 171410, 4: 2}

Two edges with four faces (solid points meeting only diagonally pinch the surface) and a separate 12-face body.

## Run 3 - 2026-11-18 15:40

Fix: before extracting, solidify one point wherever two solid (or two empty) points meet only at a cell's edge or
corner (face and cube diagonal cases, a worklist until none remain). Unit case: two cubes meeting along an edge and at a
corner; fails without the fix, passes with it.

    watertight True, volume True, bodies 2 (one of 12 faces, volume -0.0)

The second body was an inverted speck: a void the blur pinched off the bottom of a crevice. Fix: after the blur keep
the largest piece over the level and fill any empty piece not joined to the outside. Then: one body.

## Run 4 - 2026-11-18 16:00

Question: the surface shows ripples a few voxels wide (density structure, not the grid). Taubin smoothing (10 rounds,
0.5 / -0.53) helped little; blur sigma 0.8 -> 1.5 (reach 5, padding 7) removed most of it, the legs intact; extents
86.7 x 98.3 -> 87.4 x 99.1 mm (about 0.7 mm grown at 100 mm).

## Run 5 - 2026-11-18 16:20

Question: does the same sealing work on TRELLIS.2's meshes (rasterized crossings of grid edges)?

Two exported `mesh.ply` from earlier jobs (980K faces each) at 384^3: the flood leaked in, volume 0.0009 and 0.0000.
trimesh on one: 144,167 boundary edges, 2,954 boundary loops, most 3 to 8 voxels across at 384^3. A closing (grow the
occupied points 3 steps, flood, grow the flood back) sealed them: volumes 0.0359 and 0.0353.

## Run 6 - 2026-11-18 16:40

A stone well (a TRELLIS.2 result from an earlier job): the closing plus a step back to put the surface between the two
shells kept only the bucket and roller (volume 0.0094). Slices of the flood: the base slab's inside was flooded, and
the tiers' walls were dashed (mesh gaps about 10 voxels). The raw mesh (25.7M faces, 1024 dual grid) did the same.

TRELLIS.2's decoder cells (`Generated::voxels`, 1024^3) as the source instead: the walls continuous in the slices, but
the base still flooded. A vertical slice showed why: the base is a hollow shell whose inner wall opens into the shaft,
so no closing short of filling the shaft seals it, and its walls (1 to 2 voxels) did not survive the blur.

## Run 7 - 2026-11-18 17:00

Fix: thicken thin parts. The opening residue (the solid minus its opening by a cube of 2 steps) grows by 2 steps; a
face-neighbour diamond flagged every convex box edge as thin and grew it (unit box 2.2 voxels oversized), a cube does
not (0.76). The well: whole, watertight, the rope and roller kept, but 2,751,188 faces (137 MB).

`cleanup::simplify` to 500K broke closure (no link condition). Added `simplify_manifold`: the same rounds, skipping an
edge whose ends share more neighbours than the third corners of its faces, stopping when only blocked collapses
remain. Well: 481,642 faces, watertight, 65.5 cm3 (65.6 before). Unit case: a double pyramid's equator edge refused.

## Run 8 - 2026-11-18 17:30

Question: what does the export cost? Robot example, `1024-cascade`: 10.7 s on the CPU (run 51.2 s against 40 s).
Stage times: flood 2.0 s, thicken 0.8 s, blur and pieces 1.0 s, island and void pieces after the blur 4.0 s (a
per-point flood over the 60M-point grid), diagonal join 0.8 s, surface 1.0 s, simplify 0.5 s.

Fix: frame the grid around the object's bounds instead of the cube, run the passes row-parallel without per-point
division, flood with coordinates on the stack. Robot: 3.1 s (run 43.6 s), 466,524 faces, watertight; the well (as
large as its cube, hollow): 12.0 s; Idol chair: 3.3 s, 122,468 faces, one body, watertight.

## Run 9 - 2026-11-18 16:30

Question: an Idol cat (a photo, `real` mode) printed "warty": bumps the size of a few Gaussians all over the body.

The thickening (Run 7) treats whatever an opening by a 2-step cube removes as thin and grows it 2 steps: on a sealed
splat surface that is every small density bump, so each one grew outward. The same cat with the thickening off for
splats: smooth, watertight, one body, 242,160 faces (260,112 before), 75.9 cm3, 2.2 s. Splats are always sealed by the
flood, so they keep no open walls to thicken; TRELLIS.2's voxel shells keep the step.

## Run 10 - 2026-11-18 16:50

Question: the same cat image in `synthetic` mode printed with small holes into the body.

Its STL was 23.3 cm3 and 453,016 faces against 75.9 cm3 in `real` mode: a hollow shell, the flood having leaked in
through a gap in the splats; the holes opened onto the hollow inside. The closing (steps grown before the flood and
back after) on that run: 1: 23.3 cm3 (hollow); 2: 79.8 cm3, 250,860 faces; 3: 79.8; 4: 79.8. Splats now close by 3
(gaps up to about 6 cells, 2.3 mm at 100 mm); the `real` run is unchanged by it (75.9 at 1, 2 and 3). Unit case: the
Gaussian sphere with a gap at its top fails at 1 and passes at 3.

## Run 11 - 2026-11-18 18:30

Question: a crouching cat photographed from above (`real`): the splat viewer showed nothing, and the STL was 11.2 cm3
against a convex hull of 172.

The viewer: Idol predicted an orthographic camera (radius 1000, fov 0.106 degrees), beyond the viewer's far plane; it
now starts any view narrower than 30 degrees at 30 degrees, from the distance that frames the same width.

The STL: the filled volume against the closing (grid points): 3: 212,948; 4: 217,710; 5: 222,946; 6: 230,740; 8: 248,377;
10: 267,689; 12: 286,498; 16: 322,483. No jump at any closing, so not a crack but a wide opening (the unseen
underside). Fill by votes instead: a free point the (grown) shell meets in at least 5 of the 6 axis directions is
inside. First try: the grown shell's own points voted too, putting back its outer layer (a unit sphere 4 voxels
oversized); then free points only, but the closing's trim of the shell's inner side left a gap and the core came out
a separate piece; then the voted core grown back across the shell by the closing; then a nub where the test sphere's
gap plugged (votes fill a narrow channel out to the grown shell's outside): the voted set is now opened by the closing
(kept where wider than twice it).

    the photo:        104.2 cm3, hull 173.1, 289,540 faces, one body, watertight
    cat, synthetic:    80.5 cm3 (79.8 before)
    cat, real:         76.4 cm3 (75.9 before)
    armchair:         321.7 cm3, 109,572 faces

Unit case: a sphere of discs open below z = -0.6 r fills (it came out hollow without the votes).

