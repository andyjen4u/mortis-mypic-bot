# Mortis MyPic Bot

Mortis 是一個自架 Discord reaction meme 機器人。它會根據使用者的訊息，
從 [Its-MyPicDB](https://github.com/Its-MyPic/Its-MyPicDB) 的字幕與動畫影格中，
找出適合用來吐槽、反諷或製造荒謬反差的圖片。

機器人只回傳圖片，字幕文字僅供內部檢索與選圖。

## 工作流程

```text
Discord /mypic
    ↓
本地 embedding 模型進行多角度候選檢索
    ↓
Qwen3 cross-encoder 將 48 張候選重排至前 5 張
    ↓
地端聊天模型從前 5 張選出最有梗的回覆
    ↓
從本地圖片快取回傳選中的 WebP
```

目前候選檢索會從四種方向取樣：

- 輕微吐槽或反諷
- 荒謬反差或故意答非所問
- 誇張、戲劇化的反應
- 朋友間欠揍、冷淡或幸災樂禍的幽默

## 功能

- 同步 Its-MyPicDB 字幕 metadata
- 可續傳、限速及有限併發的圖片下載
- SQLite、FTS5 與本地語意向量索引
- OpenAI-compatible embedding API
- llama.cpp `/v1/rerank` cross-encoder API
- OpenAI-compatible chat completions API
- llama.cpp `thinking_budget_tokens` 與 JSON Schema 輸出
- Discord `/mypic` slash command
- 同時支援 Guild Install 與 User Install
- 可在伺服器、Bot 私訊、私人及群組頻道使用
- 私訊一般文字可自動選圖；伺服器支援提及 Bot 或指定頻道自動回覆
- systemd Bot、下載與 embedding 服務範本

## 系統需求

- Python 3.9 或更新版本
- Discord Application 與 Bot Token
- OpenAI-compatible embedding endpoint
- 選配：llama.cpp-compatible reranker endpoint
- 選配：OpenAI-compatible chat completions endpoint
- 足夠儲存 Its-MyPicDB 圖片的空間

## 安裝

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
cp .env.example .env
```

填妥 `.env` 後載入環境變數：

```bash
set -a
. ./.env
set +a
```

## 準備資料

先同步 metadata 並下載少量圖片測試：

```bash
mortis-data sync --download-limit 100
```

確認來源與流程正常後下載全部圖片：

```bash
mortis-data download --all
```

建立語意向量：

```bash
mortis-data embed --batch-size 16
```

更新 embedding 模型或文件格式後，可以完整重建衍生向量：

```bash
mortis-data embed --batch-size 16 --rebuild
```

## 測試選圖

不啟動 Discord Bot 也能測試完整的檢索、cross-encoder 與 LLM 重排：

```bash
mortis-data query "今天加班到快死了" \
  --limit 5 --cross-rerank --rerank
```

目前部署使用 `Qwen3-Reranker-0.6B Q8_0 GGUF`，固定放在 GTX 1060，
服務範本是 [`deploy/qwen3-reranker.service`](deploy/qwen3-reranker.service)。
模型來源 revision 為 `a02f48bb4f057028298c21fa033da2b30d7742d5`，
GGUF SHA-256 為
`22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48`。

## 啟動 Bot

```bash
mortis-bot
```

Rocky Linux 等 systemd 環境可參考 [`deploy/`](deploy/) 內的服務範本。

## 自動回覆

自動回覆預設開啟，行為如下：

- 私訊 Mortis：每則一般文字訊息都會自動選圖回覆。
- 伺服器：預設必須提及 Mortis。
- `AUTO_REPLY_CHANNEL_IDS` 中的頻道：不需提及 Mortis。
- Bot 與 Webhook 訊息一律忽略，避免無限回覆。
- 同一使用者及頻道預設有 10 秒冷卻時間。

```dotenv
AUTO_REPLY_ENABLED=true
AUTO_REPLY_DMS=true
AUTO_REPLY_GUILD_MENTIONS_ONLY=true
AUTO_REPLY_CHANNEL_IDS=123456789012345678,234567890123456789
AUTO_REPLY_COOLDOWN_SECONDS=10
```

自動讀取一般訊息需要在 Discord Developer Portal 的 Bot 設定中啟用
**Message Content Intent**。

## 分析選圖原因

每次選圖會在本機追加一筆 JSONL 決策紀錄，預設位置是：

```text
/var/lib/mortis-bot/decisions.jsonl
```

每筆紀錄包含：

- 使用者原始訊息與 Discord 訊息／頻道識別碼
- 四個語意檢索查詢
- 48 張召回候選及 cross-encoder 排序後的前 5 張
- 候選字幕、segment ID、語意相似度、檢索角度及 reranker 分數
- 地端聊天模型最後選擇的候選編號
- 模型提供的簡短理由、幽默手法與信心值
- 最後傳送的本機圖片路徑

這是可供稽核的決策摘要，不是模型不可驗證的內部逐步思考。紀錄包含使用者
訊息，Linux 上會以 `0600` 權限建立；如不需要可設定
`DECISION_LOG_ENABLED=false`，或以 `DECISION_LOG_PATH` 指定其他位置。

查看最新一筆：

```bash
tail -n 1 /var/lib/mortis-bot/decisions.jsonl | python -m json.tool
```

## Discord 安裝模式

Discord Developer Portal 的 Installation 設定需同時啟用：

- User Install：`applications.commands`
- Guild Install：`applications.commands`、`bot`

`/mypic` 是 global command，支援 Guild Install、User Install，以及 Guild、
Bot DM 和 Private Channel contexts。

## 安全注意事項

- 不要提交 `.env`、Discord Token、API Key 或 SSH 私鑰。
- 正式環境建議將環境檔設為只有服務帳號或 root 可以讀取。
- `DATA_DIR`、圖片、SQLite 資料庫與向量索引都不應提交至 Git。
- 圖片與字幕資料的使用條款以 Its-MyPicDB 及其上游來源為準。
