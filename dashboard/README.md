# Mortis Decision Observatory

Mortis 的本機決策 Dashboard。由 `mortis-dashboard` 直接提供 CT108 最近的
`decisions.jsonl`，可逐筆查看群聊語境、Planner 原始與校正後輸出、召回
查詢、候選分數、最終選圖、原始圖片與各階段耗時。

資料只在 CT108 與使用者的瀏覽器之間傳輸，不會送到外部服務。仍可手動匯入
JSONL 進行離線分析。

```bash
npm install
npm run dev
npm test
```
