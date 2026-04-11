# GEX 託管服務實作計畫

## 目的

本文件提出一份面向正式上線的實作計畫，目標是把目前本地端、以 Dash 為基礎的 GEX 監控工具，演進為可部署在 `gex.my-domain.com` 的網頁服務，並支援由主網域會員系統控管的多使用者存取。

本文件的用途是讓團隊先進行架構與範圍審查，再決定後續工程動作，因此內容會刻意寫得較完整，方便產品、工程與營運一起評估架構、時程、風險與優先順序。

---

## 1. 執行摘要

### 現況

目前的 repository 本質上是一個本地端 Python 應用，主要特徵如下：

- Dash UI 與 Python 程序綁在一起
- app worker 直接連線到 IB Gateway / TWS
- 以 Parquet 作為本地檔案儲存
- app 本身沒有獨立的帳號系統
- 沒有正式的 production web deployment 架構

這樣的設計很適合單人本地使用，但還不是多使用者、由會員系統控管、可對外託管服務的正確形態。

### 目標狀態

我們要把系統改造成一個具備以下能力的 web platform：

- 支援由主網域會員系統控管的存取驗證
- 後端集中式市場資料擷取
- 中央化的 GEX 計算與儲存
- 提供目前值與歷史回放的 API
- 在 `gex.my-domain.com` 提供瀏覽器前端
- 部署在 DigitalOcean

### 已確認的技術決策

團隊已確認目標技術棧如下：

- Frontend: Next.js
- Frontend styling: Tailwind CSS
- Backend API: FastAPI

因此本文件應被視為一份 Next.js + FastAPI 的實作計畫，而不是仍在比較框架方向的文件。

### 核心架構原則

即時市場資料與 GEX 應先由後端集中蒐集與計算，再透過 API 與 realtime channel 分發給已通過驗證的使用者。使用者的瀏覽器 session 不應直接連到 IB Gateway。

### 身分邊界

`gex.my-domain.com` 不應自行處理 signup、login、membership 或 billing。這些責任應保留在主網域會員網站。GEX 服務只需要驗證主網域帶來的 session 或 token，再決定是否放行。

### Repository Constraint

目前 web 實作必須先放在這個 repository 內，但又要與現有 desktop tool 明確隔離。短期目標是平行演進，而不是大規模入侵式重構。

這代表：

- 新的 web application 應放在如 `web/` 這種獨立資料夾
- 現有 desktop-oriented Python tool 應維持可持續演進，不被結構性破壞
- shared logic 只有在 reuse boundary 明確時才抽取
- 未來若要獨立成單獨 repo，或未來要合併回主站 repo，都應保留彈性

---

## 2. 產品目標

### 主要目標

- 把服務正式託管於 `gex.my-domain.com`
- 需要持有主網域有效會員 session 才能使用受保護功能
- 支援多位使用者同時在線
- 在瀏覽器提供即時 GEX 與歷史回放
- 保存 GEX 專屬的使用者資料，例如標註、提醒與偏好設定

### 次要目標

- 建立可維運的 production deployment 流程
- 降低系統對本地桌面環境的依賴
- 讓 codebase 更容易演進成 SaaS 產品

### Phase 1 非目標

- 訂閱收費與 billing
- 手機 app
- 複雜的租戶管理
- 完整的券商下單或交易執行

---

## 3. 為什麼目前的 App 不能直接部署上線

目前 codebase 雖然是很好的 prototype，但若直接做成 hosted service，會有幾個結構性問題：

- Dash UI 與應用程式 runtime 綁在一起，不利於前端、API 與 worker 的職責分離。
- 系統假設由 app 本身直接連線到 IB Gateway/TWS，這不適合多使用者的公開託管架構。
- 使用本地 Parquet 檔案作為主要資料來源，不適合多使用者查詢、帳號隔離與 production 穩定性。
- 尚未與主網域會員/session 系統整合。
- 缺少 reverse proxy、TLS、background worker、managed storage 與 observability 等 production 基礎設施。

---

## 4. 建議的目標架構

### 4.1 高階元件

建議把系統拆成以下幾個明確的部分：

1. **Web Frontend**
   - 提供 dashboard、回放、標註等瀏覽器 UI
   - 對外網址為 `gex.my-domain.com`

2. **API Service**
   - 提供受保護的 HTTP API，處理即時 GEX、歷史資料、使用者設定與分段標註
   - 可支援 WebSocket 或 Server-Sent Events

3. **市場資料擷取 Worker**
   - 連線到 IB 或其他資料來源
   - 標準化市場資料
   - 計算 GEX 與衍生特徵
   - 寫入資料庫與快取

4. **Database**
   - 儲存身份映射、snapshot、strike-level 資料、標註、提醒與審計資料

5. **Realtime Distribution Layer**
   - 把近即時更新推送給已通過驗證的瀏覽器客戶端

6. **Reverse Proxy / Edge Layer**
   - 處理 TLS termination
   - 路由 frontend 與 API
   - 綁定 `gex.my-domain.com`

### 4.2 建議技術選型

#### 已確認的技術棧

- Frontend: Next.js
- Frontend styling: Tailwind CSS
- API: FastAPI
- Database: Managed PostgreSQL
- Realtime: WebSocket 或 SSE
- Background jobs: Python worker service
- Deployment: DigitalOcean App Platform 或 Droplet
- Access control: 由主網域委派的 membership/session 驗證機制

### 前端實作方向

前端應以 Next.js 搭配 Tailwind CSS 作為主要實作方式。

這代表：

- protected pages 應以 route-level gating 為核心
- UI 應由可重用 components 組成，而不是延續 Dash 式頁面拼裝
- design tokens 應集中在 Tailwind config 與 app theme variables
- 圖表與 replay 控制應以產品化的瀏覽器 UI 方式實作

### Repository Organization Guidance

為了避免污染目前 desktop codebase，建議的近期結構如下：

```text
gex/
├── src/                      # existing desktop/local Python app
├── web/                      # new Next.js app and web-track docs
├── services/
│   └── worker/               # extraction target for ingestion logic
└── config/
```

在這個階段：

- `src/gex_monitor/...` 仍是 desktop tool 路徑
- `web/` 內含所有 Next.js 與 Tailwind 資產
- backend/API work 可在 `services/api/` 開始，或保留到 Phase 1 再展開
- cross-cutting refactor 應盡量延後，避免在 hosted architecture 尚未驗證前擴散

---

## 5. 資料流模型

### 建議流程

1. Worker 連到上游市場資料來源
2. Worker 抓取 spot、option chain、OI、greeks、expiry 等資訊
3. Worker 計算 GEX、walls、flip、regime 特徵與其他衍生分析
4. Worker 將即時 snapshot 與歷史寫入資料庫
5. API 讀取整理好的資料提供給瀏覽器
6. Realtime layer 將更新推送給已通過會員驗證的使用者
7. 使用者自己的標註與提醒則獨立儲存，在前端畫面中疊加

### 共享資料與使用者專屬資料

#### 共享市場資料

- 現價 spot
- total GEX
- gamma flip
- call wall / put wall
- strike-level GEX 分布
- 時間序列歷史
- replay snapshots

#### 使用者專屬資料

- 主網域會員身份映射與存取上下文
- 個人版面配置
- watchlist
- alerts
- 手動 regime 標註
- 私人筆記

這個切分對系統效能與產品設計都非常重要。

---

## 6. 最關鍵限制：市場資料授權與再分發

這是整份計畫中最重要的非技術問題。

若系統從 IB 取資料，再把即時或近即時的衍生市場資訊分發給多位使用者，團隊必須先確認：

- Interactive Brokers 的使用條款
- 交易所 market data licensing 規範
- 衍生計算結果是否可被視為可再分發資料
- 延遲資料與即時資料在規範上是否有差異
- 商業化與多使用者是否會觸發額外義務

### 建議

在正式投入公開多使用者產品開發之前，請先安排一次短但嚴謹的 legal/commercial review，確認這種產品模式是否被允許。

這件事應在正式實作前完成。

---

## 7. 建議的分階段實作

### 跨子網域 Session 前提

本計畫假設主網域會員網站能以以下其中一種方式供 `gex.my-domain.com` 使用：

- 一個可供 `.my-domain.com` 共用的 parent-domain cookie，或
- 一個可由 GEX 後端進行 server-side 驗證的 signed token 或 session introspection 機制

這個機制必須在實作前先確認，因為 cookie scope、`SameSite`、簽章驗證、過期、refresh 與 logout invalidation 都會直接影響最終設計。

### Phase 0：架構與合規審查

#### 目標

- 確認目標架構
- 對齊 v1 產品範圍
- 驗證市場資料授權假設
- 確認主網域 membership 整合模型

#### 交付物

- 核准的 architecture diagram
- 核准的 v1 feature scope
- data-source compliance decision memo
- 核准的跨子網域 session 設計

#### 結束條件

- 團隊對於 IB 是否可作為 hosted redistribution 的來源已有結論
- 團隊對於 `gex.my-domain.com` 如何驗證主網域會員已取得一致結論

---

### Phase 1：後端基礎建設

#### 目標

- 建立 production-friendly backend 骨架
- 引入持久化資料庫
- 定義 access-control 與 API 邊界

#### 工作項目

- 建立 FastAPI service
- 設計 database schema
- 串接主網域 membership/session 驗證
- 實作 health、current snapshots、history、user profile 等 API
- 定義 development、staging、production 的設定模型

#### 建議核心資料表

- `users` 或 `member_identities`
- `membership_context` 或外部 session mapping
- `symbols`
- `gex_snapshots`
- `gex_strikes`
- `ohlc_bars`
- `segments`
- `alerts`
- `user_preferences`

#### 結束條件

- 持有主網域有效會員 session 的使用者可進入 app
- 受保護 API 僅在會員驗證通過時可成功存取
- backend 可正常對 PostgreSQL 進行讀寫

---

### Phase 2：擷取 Worker 抽離

#### 目標

- 把市場資料擷取從 user-facing web layer 分離出去
- 集中化 GEX 計算流程

#### 工作項目

- 將既有 IB worker 邏輯重構成獨立 worker service
- 把計算邏輯整理成可重用的 service modules
- 將結果寫入 database，而非只寫本地 Parquet
- 可保留 Parquet 作為內部備份或 archive 匯出格式
- 補強 reconnect、worker health monitoring、失敗恢復機制

#### 備註

目前這些模組中的邏輯大多可以保留並重組，而不一定需要從零重寫：

- `src/gex_monitor/ib_client.py`
- `src/gex_monitor/gex_calc.py`
- `src/gex_monitor/features.py`
- `src/gex_monitor/time_utils.py`

#### 結束條件

- worker 可以獨立運作
- 即時市場資料已集中寫入
- frontend/API 不需直接存取 IB

---

### Phase 3：Web Frontend

#### 目標

- 以瀏覽器導向、由會員控管存取的 web app，取代或超越目前的 Dash UI

#### 工作項目

- 實作會員 gated access flow
- 實作 dashboard shell
- 實作即時 GEX 視圖
- 實作歷史回放
- 實作使用者標註與偏好
- 實作 realtime 更新訂閱

#### UX 指導原則

前端不應只是薄薄地複製目前 Dash layout，而應設計成真正的瀏覽器產品，具備：

- responsive layout
- membership-aware navigation
- 清楚的 loading 與 stale-data 狀態
- protected routes
- 使用者設定與提醒管理

#### 結束條件

- 持有有效主網域會員身份的使用者可從瀏覽器進入 dashboard
- 即時與歷史資料都由 API 載入
- 瀏覽器 session 不依賴直接 broker connection

---

### Phase 4：在 DigitalOcean 上線

#### 目標

- 正式把服務部署到 `gex.my-domain.com`

#### 部署選項

##### Option A：DigitalOcean App Platform

優點：

- 部署與 routing 較簡單
- TLS 與 domain 設定較方便
- 適合 frontend、API、worker 分離部署

缺點：

- 對特殊 broker networking 或 runtime 客製化的控制較少

##### Option B：DigitalOcean Droplet

優點：

- 對 runtime 與網路有完整控制權
- 比較適合客製 reverse proxy、worker process 與特殊部署需求

缺點：

- 維運負擔更高

#### 建議

若上游資料來源的連線模式允許，App Platform 對 frontend/API 會比較乾淨。若 broker 連線需要較特殊的 host-level 控制，則可能必須使用 Droplet。

#### 工作項目

- 建立 production 環境
- 配置 secrets 與 environment variables
- 配置 managed database
- 設定 domain 與 TLS
- 建立 deployment pipeline
- 建立 health check 與 restart policy

#### 結束條件

- 可透過 `https://gex.my-domain.com` 存取
- membership-gated access 可正常運作
- backend 與 worker 狀態健康
- production monitoring 已建立

---

## 8. 建議 API 介面

初始 API 可考慮：

- `GET /session`
- `GET /me`
- `GET /health`
- `GET /symbols`
- `GET /snapshots/current`
- `GET /snapshots/{symbol}/history`
- `GET /strikes/{symbol}/latest`
- `GET /replay/{symbol}`
- `GET /segments`
- `POST /segments`
- `PUT /segments/{id}`
- `DELETE /segments/{id}`
- `GET /alerts`
- `POST /alerts`

正式環境建議統一掛在 `/api/v1/...` 下。

---

## 9. Membership 整合與權限控制

### 必要條件

- 沒有主網域有效會員 session 的使用者不能讀取受保護資料
- 每位使用者都必須可映射到穩定的 membership identity
- 使用者專屬資料必須做權限驗證
- session lifecycle 必須安全

### 初始建議功能

- 主網域 session 驗證 middleware
- 安全的跨子網域 cookie 或 signed token 驗證
- route protection
- 當會員不存在或已過期時，提供清楚的 redirect 或 deny flow

### 未來延伸

- role-based access
- organization / team account
- membership tier gating

### 整合說明

建議的流程是：

1. 主網域會員網站先完成使用者驗證
2. 主網域會員網站發出可供子網域使用的 session artifact
3. `gex.my-domain.com` 在 server-side 驗證該 artifact
4. GEX 服務建立或更新本地 identity mapping
5. 所有受保護 API 與 realtime request 都依據此驗證結果放行

如果主網域只提供瀏覽器 cookie，而沒有驗證 endpoint，則團隊必須進一步定義：

- cookie domain
- cookie signing 或 encryption scheme
- expiry 與 refresh 行為
- 跨子網域 logout invalidation 行為
- server-side verification contract

---

## 10. 資料庫與儲存策略

### 建議的 production 儲存

- PostgreSQL 作為應用資料與歷史 snapshot 主儲存
- Redis 可選，用於快取與 realtime fanout
- object storage 可選，用於 archive 與匯出

### 遷移建議

現有 Parquet 儲存可以暫時保留在以下用途：

- archive export
- debugging
- offline analysis

但提供給受 membership 保護的使用者請求之 production 讀取路徑，應逐步改成 database-backed access。

---

## 11. Realtime 傳輸策略

### 可選方案

- WebSocket：若 UI 需要較複雜的雙向互動
- Server-Sent Events：若主要是單向由 server 推播更新給瀏覽器

### 初始建議

若 UI 只需要接收即時更新，建議先用 SSE，營運與實作都較簡單。若之後互動需求更複雜，再升級到 WebSocket。

---

## 12. 安全需求

最少應具備：

- 全站 HTTPS
- 安全的 secret management
- 安全的跨子網域 session 處理
- 必要時的 CSRF 防護
- session validation endpoint rate limiting
- 關鍵帳號操作的 audit logging
- public endpoint 與 internal endpoint 的清楚區隔

---

## 13. 監控與營運

### 必備 production 指標

- API health
- worker health
- database health
- ingestion lag
- 每個 symbol 最後一次成功更新時間
- membership verification error
- page/API latency

### 建議工具

- structured logs
- uptime checks
- metrics dashboard
- error tracking

---

## 14. 測試策略

### Unit Tests

- GEX 計算
- feature derivation
- config parsing
- membership verification helpers
- permission checks

### Integration Tests

- API endpoints
- database writes/reads
- worker-to-database flow
- realtime subscriptions

### End-to-End Tests

- membership-gated entry flow
- dashboard rendering
- historical replay
- segment annotation flow

---

## 15. 建議的 Repository 演進方式

### 建議結構

```text
gex/
├── apps/
│   ├── web/                  # Next.js frontend
│   └── api/                  # FastAPI service
├── services/
│   └── worker/               # market data ingestion and GEX computation
├── packages/
│   └── core/                 # shared domain logic / calculation modules
├── config/
├── web/
│   └── docs/
└── infra/
```

### 近期工作結構

由於目前的要求是讓 hosted web work 先被隔離在 repository 內，同時讓 desktop tool 繼續平行演進，因此建議的 immediate layout 是：

```text
gex/
├── src/
│   └── gex_monitor/          # current desktop/local app
├── web/                      # Next.js + Tailwind app
│   └── docs/                 # web-track planning and design docs
├── services/
│   ├── api/                  # FastAPI service
│   └── worker/               # ingestion and computation
├── config/
└── infra/
```

這樣做的好處是：

- 新的 web work 被明確隔離
- 目前 desktop code 的變動最小
- 未來若 `web/` 要搬去新 repo，路徑清楚
- 未來若 `web/` 要合併回主站 repo，整合也清楚

### 較長期的替代結構

如果之後團隊要做更正式的 monorepo split：

```text
gex/
├── apps/
│   ├── web/
│   └── api/
├── services/
│   └── worker/
├── packages/
│   └── core/
├── config/
├── web/
│   └── docs/
└── infra/
```

兩種都可行，但以目前條件來看，近期工作結構更適合。

---

## 16. 風險

### 技術風險

- IB 在 cloud environment 的連線模式可能較難維運
- 大量 option chain 擷取在規模化時可能會慢或昂貴
- strike-level 歷史資料量可能快速膨脹
- 若 realtime payload 設計不佳，fanout 成本會很高

### 產品風險

- 分析平台與交易平台的產品邊界不清楚
- 架構未穩定前，功能膨脹太快

### 合規風險

- 衍生市場資料的再分發權利
- 交易所授權義務

### 營運風險

- market hours 中 worker 發生故障
- 資料已 stale 但 UI 沒有清楚提示

---

## 17. 建議的近期下一步

1. 先讓團隊審查並核准本文件。
2. 先對資料來源與合規做出結論。
3. 決定初版 realtime transport 採用 SSE 還是 WebSocket。
4. 核准 Phase 1 的 membership verification、API 與 database schema technical design。
5. 核准初版 Next.js frontend shell 與 Tailwind design system 方向。
6. 在架構與合規對齊之前，不建議直接開做 production 實作。

---

## 18. 建議結論

### 建議方向

若目標是真正的 hosted product，正確方向應是：

- 中央化 ingestion worker
- 中央化 database
- 由 membership 控管的 web frontend
- API 驅動資料交付
- DigitalOcean 上的正式部署

在已確認的技術棧下，這代表：

- Next.js frontend
- Tailwind CSS styling system
- FastAPI backend
- Python worker service 負責 ingestion 與 computation

目前的 Dash app 應被視為原型與可重用的 domain logic 來源，而不是最後的 production application。
