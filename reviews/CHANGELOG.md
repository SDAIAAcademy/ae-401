# Changes

## 2027.2.0 (2026-11-24)

- Added: stage timings of a photo question read from the client's clock, where the upload of a full-size photo dominates the time to first token, and the fix of resizing on the phone confirmed by a rerun: `photo-upload-dominates`, `resize-on-the-phone`, and a parallel case where the queue dominates instead, `same-table-other-decision` (day 1, session 3).
- Added: the script that computes these tables, `labs/_tools/stage_timings.py`.
- Added: a practice question on photo upload time, `q-lo3-photo-upload`.
- Reworded: `three-clocks` (time to first token includes any upload); tightened `closed-loop-users` and the notes of `measure-from-the-stream` (day 1).
