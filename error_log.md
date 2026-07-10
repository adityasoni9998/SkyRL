# Why you should not store Tinker DB on NFS partition (by Codex)
Tue Jul 7 12:22:16 wekafsio:
ERROR:: wekafsio: poison file dverf change!!!
N[logs_2.sqlite-shm] sz=0x8000
weka_pages_deferred_writeout

What it means at a high level:

logs_2.sqlite-shm is a SQLite shared-memory sidecar file, usually created when SQLite uses WAL mode. WekaFS was trying to write back cached/dirty pages for that file later via
deferred writeout. While doing so, it detected the file’s version changed underneath the writeback state: cur-VC ... writeout-dverf ... mismatch. WekaFS marked/logged this as a
“poison file dverf change” writeout consistency problem.

Why this matters:

SQLite *.sqlite-shm files are coordination files. They are mmap’d and used for locks/index state between processes. Network/distributed filesystems can be problematic for SQLite
WAL/shared-memory semantics. If WekaFS has delayed writeback or version-conflict behavior, processes doing SQLite reads/writes can stall in filesystem syscalls, mmap page faults,
fsync, lock acquisition, or retry paths.

What it does not show:

It does not show OOM, GPU failure, NCCL/RCCL failure, or Ray actor death. It also does not name tinker.db directly. The file is logs_2.sqlite-shm, so the direct affected DB
appears to be some logging SQLite DB, not necessarily the Tinker FutureDB.

Could it cause stalls?

Yes, if the process path is touching that DB or the same WekaFS mount heavily. It could stall logging, telemetry, SQLite-backed request/future state, or any code blocked on DB/
file locks. But the logs show vLLM was still serving requests after 12:22:16, so this is not evidence that all engines froze.

Practical implication: avoid SQLite WAL/shared-memory files on WekaFS for this workload. Put SQLite DBs and logs on local disk, ideally /tmp, and only copy artifacts back later.
This is especially relevant for Tinker DB/FutureDB or any high-frequency logging DB.

# Fix KL Logging Change Explanation (by Codex)

## Short answer

Yes, SkyRL could compute KL-side metrics internally and return them to Platoon, but that is not the minimal fix.

The minimal fix is to keep training exactly as-is and only reorder the returned `loss_fn_outputs` in SkyRL's FSDP `forward_backward` path when `max_tokens_per_microbatch > 0`.

## Why Platoon currently fails

Platoon assumes:

- `forward_backward_result.loss_fn_outputs[i]` corresponds to `datums[i]`
- each returned `logprobs` vector matches that datum's mask length

With token-based microbatching, SkyRL bin-packs samples into microbatches and returns `loss_fn_outputs` in packed microbatch order instead of original sample order. Platoon then applies the wrong mask to the wrong sample's logprobs, so KL computation fails.

## Why not compute KL directly in SkyRL

Possible, but not minimal:

- SkyRL would need to either reproduce Platoon's KL metric semantics exactly or return a new metric contract
- it would push Platoon-specific logging logic into SkyRL
- it still would not fix the broken per-sample API contract for other consumers

So computing KL inside SkyRL is a larger behavioral change than necessary.

## Minimal fix location

File:

- `skyrl/backends/skyrl_train/workers/worker.py`

Function:

- `Worker.forward_backward(...)`

## Minimal fix shape

Current pattern:

```python
all_loss_fn_outputs = []

for microbatch in microbatch_iterator:
    ...
    if "loss_fn_outputs" in metrics:
        all_loss_fn_outputs.extend(metrics.pop("loss_fn_outputs"))

return WorkerOutput(loss_fn_outputs=all_loss_fn_outputs, metrics=result)
```

Replace with:

```python
loss_fn_outputs_by_microbatch = []

for microbatch in microbatch_iterator:
    ...
    if "loss_fn_outputs" in metrics:
        loss_fn_outputs_by_microbatch.append(metrics.pop("loss_fn_outputs"))
    else:
        loss_fn_outputs_by_microbatch.append([])

if self.cfg.max_tokens_per_microbatch > 0:
    reordered = [None] * len(data)

    for mb_outputs, original_indices in zip(
        loss_fn_outputs_by_microbatch,
        microbatch_iterator._microbatches,
    ):
        if len(mb_outputs) != len(original_indices):
            raise ValueError(
                f"loss_fn_outputs length mismatch: {len(mb_outputs)} outputs "
                f"for {len(original_indices)} samples"
            )
        for output, original_idx in zip(mb_outputs, original_indices):
            reordered[original_idx] = output

    if any(x is None for x in reordered):
        raise ValueError("Missing loss_fn_outputs after token-based reorder")

    all_loss_fn_outputs = [x for x in reordered if x is not None]
else:
    all_loss_fn_outputs = [
        output
        for mb_outputs in loss_fn_outputs_by_microbatch
        for output in mb_outputs
    ]

return WorkerOutput(loss_fn_outputs=all_loss_fn_outputs, metrics=result)
```

## Why this is safe

This changes only the returned per-sample output ordering.

It does **not** change:

- microbatch packing
- forward/backward execution
- loss computation
- gradient accumulation
- optimizer stepping
- metric reduction

So it fixes KL logging without changing training behavior.

## Optional validation

After the SkyRL fix, Platoon should again satisfy:

- `len(forward_backward_result.loss_fn_outputs) == len(original_datums)`
- for each datum/output pair, `len(output["logprobs"]) == datum.loss_fn_inputs["mask"].sum()`

