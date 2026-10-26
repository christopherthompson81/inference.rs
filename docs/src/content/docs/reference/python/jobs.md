---
title: Jobs
description: "World generation and reconstruction jobs: what to run, their records, stages, previews, outputs and events."
sidebar:
  order: 8
---
## `JobDeleted`

A job removed with its folder.

| Field | Type |
| --- | --- |
| `deleted` | `bool` |
| `id` | `str` |


## `JobEvent`

One of: `Union[JobEventStatus, JobEventWaiting, JobEventStage, JobEventProgress, Any, JobEventFinished]`.


## `JobEventFinished`

| Field | Type | Default |
| --- | --- | --- |
| `error` | `str \| None` | optional |
| `status` | `JobStatus` | required |
| `type` | `Literal['finished']` | `'finished'` |


## `JobEventProgress`

| Field | Type | Default |
| --- | --- | --- |
| `done` | `int` | required |
| `total` | `int` | required |
| `type` | `Literal['progress']` | `'progress'` |


## `JobEventStage`

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | required |
| `total` | `int` | required |
| `type` | `Literal['stage']` | `'stage'` |


## `JobEventStatus`

| Field | Type | Default |
| --- | --- | --- |
| `status` | `JobStatus` | required |
| `type` | `Literal['status']` | `'status'` |


## `JobEventWaiting`

A queued job can't start yet, and why.

| Field | Type | Default |
| --- | --- | --- |
| `reason` | `str` | required |
| `type` | `Literal['waiting']` | `'waiting'` |


## `JobFilePath`

Where a job's file is on this machine.

| Field | Type |
| --- | --- |
| `path` | `str` |


## `JobFileRequest`

A file in a job's folder, given relative to it.

| Field | Type |
| --- | --- |
| `file` | `str` |
| `id` | `str` |


## `JobList`

This owner's jobs, newest first.

| Field | Type |
| --- | --- |
| `data` | `list[JobRecord]` |


## `JobOutput`

A file a job made.

| Field | Type |
| --- | --- |
| `file` | `str` |
| `name` | `str` |


## `JobPreview`

An image that shows how the job is going.

| Field | Type |
| --- | --- |
| `caption` | `str` |
| `file` | `str` |
| `stage` | `str` |


## `JobRecord`

A job's record: what it was asked, how far it got, and what it made.

| Field | Type | Default |
| --- | --- | --- |
| `created_at` | `float` | required |
| `device` | `int \| None` | optional |
| `error` | `str \| None` | optional |
| `finished_at` | `float \| None` | optional |
| `id` | `str` | required |
| `inputs` | `list[str] \| None` | optional |
| `kind` | `str` | required |
| `outputs` | `list[JobOutput] \| None` | optional |
| `owner` | `str \| None` | optional |
| `previews` | `list[JobPreview] \| None` | optional |
| `report` | `dict[str, Any] \| None` | optional |
| `request` | `dict[str, Any]` | required |
| `stages` | `list[JobStage] \| None` | optional |
| `started_at` | `float \| None` | optional |
| `status` | `JobStatus` | required |
| `waiting` | `str \| None` | optional |


## `JobStage`

| Field | Type | Default |
| --- | --- | --- |
| `done` | `int` | required |
| `finished_at` | `float \| None` | optional |
| `name` | `str` | required |
| `started_at` | `float` | required |
| `total` | `int` | required |


## `JobStatus`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `JobStatus.QUEUED` | `'queued'` |
| `JobStatus.RUNNING` | `'running'` |
| `JobStatus.SUCCEEDED` | `'succeeded'` |
| `JobStatus.FAILED` | `'failed'` |
| `JobStatus.CANCELLED` | `'cancelled'` |


## `JobSubmission`

A job and the files it takes, by their paths on this machine.

| Field | Type | Default |
| --- | --- | --- |
| `inputs` | `list[str] \| None` | optional |
| `request` | `JobRequest` | required |


## `JobsConfig`

Where jobs keep their folders and which weights they run.

| Field | Type | Default |
| --- | --- | --- |
| `devices` | `list[int] \| None` | optional |
| `dir` | `str \| None` | `'/home/chris/.cache/inference-rs/engine-jobs'` |
| `drain_timeout_secs` | `int \| None` | `120` |
| `lpips` | `str \| None` | optional |
| `moge` | `str \| None` | optional |
| `placement` | `Placement \| None` | `Placement.FASTEST` |
| `wan` | `str \| None` | optional |
| `worldmirror` | `str \| None` | optional |
| `worldstereo` | `str \| None` | optional |


## `Placement`

How much a job may displace to run.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `Placement.FASTEST` | `'fastest'` |
| `Placement.WHEN_QUIET` | `'when_quiet'` |
| `Placement.NO_EVICTION` | `'no_eviction'` |


## `ReconstructJob`

WorldMirror 2.0 on the job's inputs: images, or one video.

| Field | Type | Default |
| --- | --- | --- |
| `fps` | `float \| None` | optional |
| `max_frames` | `int \| None` | optional |
| `placement` | `Placement \| None` | optional |
| `target_size` | `int \| None` | optional |


## `SceneKind`

Indoor scenes keep far distances; outdoor ones compress them.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `SceneKind.INDOOR` | `'indoor'` |
| `SceneKind.OUTDOOR` | `'outdoor'` |


## `WorldJob`

A world from a panorama, the job's one input.

| Field | Type | Default |
| --- | --- | --- |
| `placement` | `Placement \| None` | optional |
| `prompt` | `str \| None` | optional |
| `scene` | `SceneKind \| None` | optional |
| `seed` | `int \| None` | optional |
| `splats` | `bool \| None` | optional |
| `steps` | `int \| None` | optional |

---

<small>Generated from [`bindings/python/inference_rs`](https://github.com/christopherthompson81/inference.rs/blob/master/bindings/python/inference_rs).</small>
