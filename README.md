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
地端聊天模型依「哪張最有梗」重排候選
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
- OpenAI-compatible chat completions API
- llama.cpp `thinking_budget_tokens` 與 JSON Schema 輸出
- Discord `/mypic` slash command
- 同時支援 Guild Install 與 User Install
- 可在伺服器、Bot 私訊、私人及群組頻道使用
- systemd Bot、下載與 embedding 服務範本

## 系統需求

- Python 3.9 或更新版本
- Discord Application 與 Bot Token
- OpenAI-compatible embedding endpoint
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

不啟動 Discord Bot 也能測試完整的檢索與 LLM 重排：

```bash
mortis-data query "今天加班到快死了" --limit 12 --rerank
```

## 啟動 Bot

```bash
mortis-bot
```

Rocky Linux 等 systemd 環境可參考 [`deploy/`](deploy/) 內的服務範本。

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
