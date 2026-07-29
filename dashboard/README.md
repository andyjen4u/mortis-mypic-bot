# Mortis Decision Observatory

Mortis 的本機優先決策 Dashboard。匯入 `decisions.jsonl` 後，可逐筆查看群聊
語境、Planner 原始與校正後輸出、召回查詢、候選分數、最終選圖與各階段耗時。

JSONL 只在瀏覽器內解析，不會由 Dashboard 上傳或保存。

```bash
npm install
npm run dev
npm test
```
