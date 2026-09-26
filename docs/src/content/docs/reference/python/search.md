---
title: Search
description: "Types for web-search tool configuration."
sidebar:
  order: 7
---
## `WebSearchOptions`

| Field | Type | Default |
| --- | --- | --- |
| `search_context_size` | `Optional[SearchContextSize]` | `None` |
| `user_location` | `Optional[WebSearchUserLocation]` | `None` |
| `search_description` | `Optional[str]` | `None` |
| `extract_description` | `Optional[str]` | `None` |


## `WebSearchUserLocation`

### `WebSearchUserLocation.approximate`

```text
approximate(
    approximate: ApproximateUserLocation,
) -> 'WebSearchUserLocation'
```


## `ApproximateUserLocation`

| Field | Type |
| --- | --- |
| `city` | `str` |
| `country` | `str` |
| `region` | `str` |
| `timezone` | `str` |

---

<small>Generated from [`crates/inference-pyo3/inference_rs.pyi`](https://github.com/christopherthompson81/inference.rs/blob/master/crates/inference-pyo3/inference_rs.pyi).</small>
