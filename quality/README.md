# Mortis quality reviews

`reviews.jsonl` 是人工／Codex 對選圖結果的增量評估紀錄。
每個 `selection_id` 只評估一次。
舊版曾強制執行 final judge 的樣本以
`pipeline: legacy_forced_final_judge` 標示，不和目前正式流程直接比較。

主要標籤：

- `natural`：圖片字幕可以自然接在訊息下方。
- `borderline`：勉強成立，但不是群友最自然的插話。
- `bad`：不適合回覆。

失敗分類：

- `should_stay_silent`
- `wrong_speaker`
- `wrong_speech_act`
- `lexical_overlap_only`
- `missing_candidate`
- `reranker_error`
- `context_error`
- `other`

評估必須同時查看最近對話、最新訊息、planner 原始與修正後輸出、
cross-encoder baseline、semantic judge 與最後字幕，不能只看關鍵字。
