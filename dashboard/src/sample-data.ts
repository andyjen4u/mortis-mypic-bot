export type DecisionRecord = {
  timestamp?: string;
  selection_id?: string;
  query?: string;
  conversation?: string;
  context?: Record<string, unknown>;
  policy?: Record<string, unknown>;
  planner?: Record<string, unknown>;
  retrieval?: Record<string, unknown>;
  candidates?: Array<Record<string, unknown>>;
  cross_encoder?: Record<string, unknown>;
  llm_reranker?: Record<string, unknown>;
  timings_ms?: Record<string, number>;
  result?: Record<string, unknown>;
};

export const sampleDecisions: DecisionRecord[] = [
  {
    timestamp: "2026-07-29T10:41:12.240Z",
    selection_id: "sample-teto",
    query: "Teto 的聲庫太強",
    conversation: "Rin：那你一定要聽聽 Science 跟 Magic Maid\nMika：我先看看有沒有翻唱",
    context: {
      trigger: "automatic_message",
      configured_mode: "off",
      evaluated_mode: "auto",
      author_name: "Andy",
      delivery_suppressed: true,
    },
    policy: {
      mode: "auto",
      activity: "medium",
      mentioned: false,
      gate_reason: "auto_threshold_passed",
    },
    planner: {
      raw: {
        reaction_goal: "用太強或厲害接住對聲庫的稱讚",
        search_terms: ["太強", "厲害", "真的"],
        meme_role: "agreement",
        speaker_perspective: "observer",
      },
      adjustments: [],
      action: "post",
      reaction_goal: "用太強或厲害接住對聲庫的稱讚",
      search_terms: ["太強", "厲害", "真的", "對啊"],
      meme_role: "agreement",
      speaker_perspective: "observer",
      reason: "最新訊息是在稱讚聲庫，適合用群友附和的角度插話。",
      confidence: 0.95,
    },
    retrieval: {
      mode: "semantic+lexical",
      lexical_terms: ["太強", "厲害", "真的", "對啊"],
      pool_size: 16,
      queries: [
        "Teto 的聲庫太強；群友自然附和",
        "用太強或厲害接住對聲庫的稱讚",
        "朋友間誇張稱讚的反應",
      ],
    },
    candidates: [
      {
        index: 0,
        segment_id: 1844,
        text: "我好像也有點太強硬了",
        semantic_score: 0.71,
        contextual_reranker_score: 0.08,
        reaction_reranker_score: 0.02,
        contextual_reranker_rank: 1,
        retrieval_pool_index: 3,
      },
      {
        index: 1,
        segment_id: 1760,
        text: "好厲害",
        semantic_score: 0.65,
        contextual_reranker_score: 0.07,
        reaction_reranker_score: 0.06,
        contextual_reranker_rank: 2,
        retrieval_pool_index: 5,
      },
      {
        index: 2,
        segment_id: 3931,
        text: "真的很強耶",
        semantic_score: 0.61,
        contextual_reranker_score: 0.05,
        reaction_reranker_score: 0.05,
        contextual_reranker_rank: 3,
        retrieval_pool_index: 8,
      },
    ],
    cross_encoder: {
      model: "qwen3-reranker-0.6b-q8_0.gguf",
      baseline: {
        index: 0,
        text: "我好像也有點太強硬了",
        perspective: "contextual",
      },
    },
    llm_reranker: {
      model: "gemma-4-12b-it-IQ4_NL",
      perspective: "disabled",
      action: "post",
      selected_index: 0,
      reason: "Final judge 未啟用，採用 cross-encoder 結果。",
      confidence: 0.95,
    },
    timings_ms: {
      queue: 0,
      planner: 3984,
      embedding: 695,
      retrieval: 730,
      reranker: 6666,
      final_judge: 0,
      image: 1,
      total: 12076,
    },
    result: {
      status: "selected",
      segment_id: 1844,
      text: "我好像也有點太強硬了",
      delivery: "suppressed_off",
    },
  },
  {
    timestamp: "2026-07-29T10:43:45.610Z",
    selection_id: "sample-silent",
    query: "這樣每一句話都要一張圖其實蠻占版面的",
    conversation: "Kiwi：我覺得不如調成指令發圖",
    context: {
      trigger: "automatic_message",
      configured_mode: "off",
      evaluated_mode: "auto",
      author_name: "Mika",
      delivery_suppressed: true,
    },
    policy: {
      mode: "auto",
      activity: "medium",
      mentioned: false,
      gate_reason: "reply_feature_feedback",
    },
    planner: {},
    retrieval: {},
    candidates: [],
    timings_ms: { queue: 0, total: 0.4 },
    result: {
      status: "stayed_silent",
      delivery: "suppressed_off",
    },
  },
  {
    timestamp: "2026-07-29T10:47:06.180Z",
    selection_id: "sample-natural",
    query: "真好聽，下次 KTV 前我再學起來",
    conversation: "Rin：這首副歌真的超洗腦\nMika：而且現場版更好聽",
    context: {
      trigger: "automatic_message",
      configured_mode: "auto",
      evaluated_mode: "auto",
      author_name: "Andy",
      delivery_suppressed: false,
    },
    policy: {
      mode: "auto",
      activity: "high",
      mentioned: false,
      gate_reason: "auto_threshold_passed",
    },
    planner: {
      raw: {
        reaction_goal: "對想學歌這件事給一個誇張但友善的稱讚",
        search_terms: ["好聽", "厲害", "期待"],
        meme_role: "agreement",
        speaker_perspective: "observer",
      },
      adjustments: [],
      action: "post",
      reaction_goal: "對想學歌這件事給一個誇張但友善的稱讚",
      search_terms: ["好聽", "厲害", "期待"],
      meme_role: "agreement",
      speaker_perspective: "observer",
      reason: "接住對方想學歌的期待，短稱讚可自然插入。",
      confidence: 0.9,
    },
    retrieval: {
      mode: "semantic+lexical",
      lexical_terms: ["好聽", "厲害", "期待"],
      pool_size: 16,
      queries: ["想學一首好聽的歌；群友稱讚", "期待下次表演"],
    },
    candidates: [
      {
        index: 0,
        segment_id: 1760,
        text: "好厲害",
        semantic_score: 0.76,
        contextual_reranker_score: 0.72,
        reaction_reranker_score: 0.81,
        contextual_reranker_rank: 1,
        retrieval_pool_index: 2,
      },
      {
        index: 1,
        segment_id: 4410,
        text: "我開始期待了",
        semantic_score: 0.7,
        contextual_reranker_score: 0.66,
        reaction_reranker_score: 0.7,
        contextual_reranker_rank: 2,
        retrieval_pool_index: 4,
      },
    ],
    cross_encoder: {
      model: "qwen3-reranker-0.6b-q8_0.gguf",
      baseline: {
        index: 0,
        text: "好厲害",
        perspective: "reaction",
      },
    },
    llm_reranker: {
      model: "gemma-4-12b-it-IQ4_NL",
      perspective: "disabled",
      action: "post",
      selected_index: 0,
      reason: "Final judge 未啟用，採用 cross-encoder 結果。",
      confidence: 0.9,
    },
    timings_ms: {
      queue: 0,
      planner: 3720,
      embedding: 620,
      retrieval: 692,
      reranker: 5920,
      final_judge: 0,
      image: 1,
      total: 10953,
    },
    result: {
      status: "selected",
      segment_id: 1760,
      text: "好厲害",
      delivery: "discord_pending",
    },
  },
];
