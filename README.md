# Mortis MyPic Bot

Mortis 是一個自架 Discord reaction meme 機器人。它會根據使用者的訊息，
從 [Its-MyPicDB](https://github.com/Its-MyPic/Its-MyPicDB) 的字幕與動畫影格中，
找出適合用來吐槽、反諷或製造荒謬反差的圖片。

機器人只回傳圖片，字幕文字僅供內部檢索與選圖。

## 工作流程

```text
Discord 訊息、提及或 /mypic
    ↓
Gemma 規劃是否插話、反應角色、核心事件詞與反應詞
    ↓
本地 embedding 與中文字詞搜尋組成 16 張混合候選
    ↓
Qwen3 cross-encoder 依反應角色做單一路重排
    ↓
字幕去重、反應詞候選保留及人物、時態、人稱 grounding filter
    ↓
直接採用角色對應的最高排名；reranker 故障時才由 Gemma 救援選圖
    ↓
從本地圖片快取回傳選中的 WebP
```

候選檢索會從五種語意方向及本地中文字詞結果取樣：

- planner 產生的具體反應目標
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
- 動態字幕搜尋詞、混合檢索、字幕去重及反應詞候選保留
- 人物、時態與人稱一致性過濾
- 可選的 Gemma 最終裁判；預設只在 reranker 故障時救援
- Planner、embedding、檢索、reranker、最終裁判與圖片準備的延遲紀錄
- Discord `/mypic` slash command
- 同時支援 Guild Install 與 User Install
- 可在伺服器、Bot 私訊、私人及群組頻道使用
- 三種回圖模式、三級積極度及分頻道／使用者持久設定
- 讀取同頻道最近 5 則訊息；提及 Bot 時一定選圖
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

插話規劃使用 RTX 2070 上的 `Gemma 4 12B IQ4_NL`，並以 `-ngl 99`
完整 offload 到 GPU；服務範本是
[`deploy/gemma-planner.service`](deploy/gemma-planner.service)。Bot 的
planner endpoint drop-in 是
[`deploy/mortis-bot-planner.conf`](deploy/mortis-bot-planner.conf)，其值放在
[`deploy/mortis-bot-planner.env`](deploy/mortis-bot-planner.env)，並在主
環境檔之後載入。`FINAL_JUDGE_ENABLED=false` 會直接採用 reranker 與
grounding filter 的結果；設為 `true` 才會對每次回圖再呼叫一次 Gemma。

若 embedding 服務因安全需求只監聽 CT102 的 `127.0.0.1:8081`，可啟用
[`deploy/nemotron-embedding-proxy.socket`](deploy/nemotron-embedding-proxy.socket)
與
[`deploy/nemotron-embedding-proxy.service`](deploy/nemotron-embedding-proxy.service)，
只在 `192.168.10.102:8081` 額外提供 systemd socket proxy 給 CT108，
不必修改模型本身的 unit。

## 啟動 Bot

```bash
mortis-bot
```

Rocky Linux 等 systemd 環境可參考 [`deploy/`](deploy/) 內的服務範本。

## 回圖模式

預設為 `auto + medium`。三種模式如下：

- `always`：每則一般文字訊息都選一張圖。
- `auto`：模型評估插話時機及候選品質後決定是否回圖。
- `off`：不監聽普通訊息。
- 無論模式為何，提及 Mortis 或使用 `/mypic` 都會強制選圖。
- Bot 與 Webhook 訊息一律忽略，避免無限回覆。
- `auto` 模式以頻道為單位套用冷卻時間；提及不受冷卻限制。
- `auto` 的 `low`、`medium`、`high` 分別使用 0.82、0.68、0.52
  的插話信心門檻。普通問候會直接保持沉默。

```dotenv
AUTO_REPLY_MODE=auto
AUTO_REPLY_ACTIVITY=medium
CONTEXT_MESSAGE_LIMIT=5
AUTO_REPLY_ENABLED=true
AUTO_REPLY_DMS=true
AUTO_REPLY_COOLDOWN_SECONDS=10
```

在 Discord 使用以下子指令查看或修改設定：

```text
/mypic-settings status
/mypic-settings always
/mypic-settings auto activity:<low|medium|high>
/mypic-settings off
```

只有 `auto` 模式需要選擇積極度。
伺服器內按頻道保存，修改需要「管理訊息」權限；私訊則按使用者保存。

自動讀取一般訊息需要在 Discord Developer Portal 的 Bot 設定中啟用
**Message Content Intent**。

## 分析選圖原因

每次選圖會在本機追加一筆 JSONL 決策紀錄，預設位置是：

```text
/var/lib/mortis-bot/decisions.jsonl
```

每筆紀錄包含：

- 使用者原始訊息、最近群聊及 Discord 訊息／頻道識別碼
- planner 的插話決策、反應角色、核心事件詞與動態字幕搜尋詞
- 五個語意檢索查詢、中文字詞候選及 16 張混合召回候選
- 候選字幕、segment ID、語意相似度、檢索角度及 reranker 分數
- planner 角色選定的單一路 reranker、反應詞保留候選及最終候選編號
- 模式、積極度、是否被提及、插話門檻結果
- 模型的 `post`／`stay_silent` 決策、簡短理由、梗圖角色與信心值
- grounding filter 與可選 Gemma 最終裁判採用的選擇觀點
- queue、planner、embedding、retrieval、reranker、final judge、圖片準備及總延遲
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
