(() => {
  "use strict";

  const STORAGE_KEY = "rag-deep-dive-learning-v1";
  const FOCUS_SECONDS = 25 * 60;

  const days = [
    { id: "d01", label: "\u8fb9\u754c\u4e0e\u5fc3\u667a\u6a21\u578b", unitIds: ["s0-u1", "s0-u2", "s0-u3", "s1-u1", "s1-u2"] },
    { id: "d02", label: "\u8bba\u6587\u4e0e\u6734\u7d20\u7ba1\u9053", unitIds: ["s1-u3", "s1-u4", "s2-u1", "s2-u2", "s2-u3"] },
    { id: "d03", label: "\u89e3\u6790\u4e0e\u5206\u5757", unitIds: ["s3-u1", "s3-u2", "s3-u3"] },
    { id: "d04", label: "Embedding \u4e0e\u53cc\u5854", unitIds: ["s4-u1", "s4-u2", "s4-u3"] },
    { id: "d05", label: "\u5411\u91cf\u68c0\u7d22\u4e0e ANN", unitIds: ["s5-u1", "s5-u2", "s5-u3"] },
    { id: "d06", label: "\u6df7\u5408\u68c0\u7d22\u4e0e\u91cd\u6392", unitIds: ["s6-u1", "s6-u2", "s7-u1", "s7-u2"] },
    { id: "d07", label: "\u67e5\u8be2\u53d8\u6362\u4e0e Self-RAG", unitIds: ["s8-u1", "s9-u1", "s9-u2"] },
    { id: "d08", label: "CRAG \u4e0e GraphRAG", unitIds: ["s9-u3", "s9-u4", "s9-u5", "s10-u1"] },
    { id: "d09", label: "Agentic RAG \u5b9e\u73b0", unitIds: ["s10-u2", "s10-u3", "s10-u4"] },
    { id: "d10", label: "\u591a\u6a21\u6001\u4e0e\u8bc4\u4f30\u6307\u6807", unitIds: ["s11-u1", "s11-u2", "s12-u1"] },
    { id: "d11", label: "\u8bc4\u4f30\u95ed\u73af\u4e0e\u751f\u4ea7\u5316", unitIds: ["s12-u2", "s13-u1", "s13-u2", "s13-u3"] },
    { id: "d12", label: "\u9762\u8bd5\uff1a\u57fa\u7840\u4e0e\u68c0\u7d22", unitIds: ["s14-u1", "s14-u2", "s14-u3", "s14-u4", "s14-u5"] },
    { id: "d13", label: "\u9762\u8bd5\uff1a\u67b6\u6784\u4e0e\u7cfb\u7edf\u8bbe\u8ba1", unitIds: ["s14-u6", "s14-u7", "s14-u8", "s14-u9", "s14-u10", "s14-u11"] },
    { id: "d14", label: "\u7efc\u5408\u9879\u76ee\u4e0e\u4e13\u5bb6\u7b54\u8fa9", unitIds: ["s15-u1", "s15-u2"] }
  ];

  const promptTemplates = {
    s0: title => "\u56f4\u7ed5\u201c" + title + "\u201d\uff0c\u7ed9\u51fa\u4e00\u53e5\u5b9a\u4e49\u3001\u4e00\u4e2a\u56e0\u679c\u94fe\uff0c\u4ee5\u53ca\u5b83\u5728\u6574\u6761 RAG \u94fe\u8def\u4e2d\u7684\u4f4d\u7f6e\u3002",
    s1: title => "\u4e0d\u770b\u6b63\u6587\u89e3\u91ca\u201c" + title + "\u201d\uff1a\u5b83\u89e3\u51b3\u4ec0\u4e48\u95ee\u9898\uff1f\u4e0e\u5fae\u8c03\u6216\u957f\u4e0a\u4e0b\u6587\u76f8\u6bd4\uff0c\u8fb9\u754c\u5728\u54ea\u91cc\uff1f",
    s2: title => "\u628a\u201c" + title + "\u201d\u653e\u8fdb\u79bb\u7ebf\u7d22\u5f15\u4e0e\u5728\u7ebf\u68c0\u7d22\u4e24\u6761\u6d41\u6c34\u7ebf\u4e2d\uff0c\u8bf4\u660e\u8f93\u5165\u3001\u8f93\u51fa\u548c\u6700\u5e38\u89c1\u5931\u8d25\u70b9\u3002",
    s3: title => "\u89e3\u91ca\u201c" + title + "\u201d\u5982\u4f55\u5f71\u54cd\u53ec\u56de\u7387\u3001\u4e0a\u4e0b\u6587\u5b8c\u6574\u6027\u4e0e\u6210\u672c\uff0c\u5e76\u8bf4\u51fa\u4f60\u4f1a\u5982\u4f55\u9a8c\u8bc1\u9009\u578b\u3002",
    s4: title => "\u4ece\u8bad\u7ec3\u76ee\u6807\u548c\u5411\u91cf\u7a7a\u95f4\u51fa\u53d1\u89e3\u91ca\u201c" + title + "\u201d\uff0c\u518d\u7ed9\u51fa\u4e00\u4e2a\u5de5\u7a0b\u9009\u578b\u5224\u65ad\u3002",
    s5: title => "\u89e3\u91ca\u201c" + title + "\u201d\u5728\u7cbe\u5ea6\u3001\u5ef6\u8fdf\u3001\u5185\u5b58\u4e09\u8005\u95f4\u505a\u4e86\u4ec0\u4e48\u53d6\u820d\u3002",
    s6: title => "\u8bf4\u660e\u201c" + title + "\u201d\u4e3a\u4ec0\u4e48\u6bd4\u5355\u4e00\u8def\u68c0\u7d22\u66f4\u7a33\uff0c\u5e76\u7ed9\u51fa\u878d\u5408\u65f6\u4e0d\u80fd\u76f4\u63a5\u6bd4\u8f83\u539f\u59cb\u5206\u6570\u7684\u539f\u56e0\u3002",
    s7: title => "\u8bf4\u660e\u201c" + title + "\u201d\u5728\u53ec\u56de\u548c\u7cbe\u6392\u4e4b\u95f4\u626e\u6f14\u4ec0\u4e48\u89d2\u8272\uff0c\u4ee5\u53ca\u5b83\u4e3a\u4f55\u4e0d\u80fd\u76f4\u63a5\u641c\u7d22\u5168\u5e93\u3002",
    s8: title => "\u4e3a\u201c" + title + "\u201d\u7ed9\u51fa\u9002\u7528\u67e5\u8be2\u3001\u6536\u76ca\u3001\u989d\u5916\u5ef6\u8fdf\u4e0e\u5931\u8d25\u8fb9\u754c\u3002",
    s9: title => "\u7528\u201c\u89e6\u53d1\u6761\u4ef6 \u2192 \u63a7\u5236\u6d41 \u2192 \u4ee3\u4ef7\u201d\u7684\u7ed3\u6784\u8bb2\u6e05\u201c" + title + "\u201d\u3002",
    s10: title => "\u628a\u201c" + title + "\u201d\u753b\u6210\u72b6\u6001\u4e0e\u6761\u4ef6\u8fb9\uff0c\u5e76\u8bf4\u660e\u4ec0\u4e48\u65f6\u5019\u56fa\u5b9a\u7ba1\u9053\u5df2\u7ecf\u8db3\u591f\u3002",
    s11: title => "\u6bd4\u8f83\u201c" + title + "\u201d\u4e2d\u7684\u6587\u672c\u5316\u3001\u5171\u4eab\u5411\u91cf\u7a7a\u95f4\u4e0e\u89c6\u89c9\u6587\u6863\u68c0\u7d22\u8def\u7ebf\u3002",
    s12: title => "\u56f4\u7ed5\u201c" + title + "\u201d\u5206\u522b\u7ed9\u51fa\u68c0\u7d22\u6307\u6807\u3001\u751f\u6210\u6307\u6807\u3001\u6570\u636e\u6765\u6e90\u548c\u56de\u5f52\u65b9\u6cd5\u3002",
    s13: title => "\u628a\u201c" + title + "\u201d\u5f53\u4f5c\u7ebf\u4e0a\u4e8b\u6545\uff0c\u6309\u89c2\u6d4b\u3001\u5f52\u56e0\u3001\u4fee\u590d\u3001\u56de\u5f52\u56db\u6b65\u4f5c\u7b54\u3002",
    s14: title => "\u6a21\u62df\u9762\u8bd5\uff0c\u4e0d\u770b\u7b54\u6848\u5148\u7528 60 \u5230 90 \u79d2\u56de\u7b54\uff1a" + title,
    s15: title => "\u56f4\u7ed5\u201c" + title + "\u201d\uff0c\u7ed9\u51fa\u53ef\u6267\u884c\u4ea7\u51fa\u3001\u8861\u91cf\u6807\u51c6\u548c\u4f60\u51c6\u5907\u5199\u8fdb\u7b80\u5386\u7684\u8bc1\u636e\u3002"
  };

  const dayBriefs = {
    d01: {
      hours: "2-2.5 \u5c0f\u65f6",
      objective: "\u5efa\u7acb\u8bc1\u636e z \u7684\u7edf\u4e00\u5fc3\u667a\u6a21\u578b\uff0c\u660e\u786e RAG\u3001\u5fae\u8c03\u4e0e\u957f\u4e0a\u4e0b\u6587\u7684\u8fb9\u754c\u3002",
      output: "\u4e00\u5f20\u9009\u578b\u51b3\u7b56\u56fe + 3 \u5206\u949f\u53e3\u8ff0\u7a3f",
      acceptance: "\u80fd\u7528\u68c0\u7d22\u5931\u8d25 / \u751f\u6210\u5931\u8d25\u6846\u67b6\u5b9a\u4f4d\u95ee\u9898\u3002"
    },
    d02: {
      hours: "2.5-3 \u5c0f\u65f6",
      objective: "\u8bfb\u61c2\u539f\u59cb RAG \u7684\u6982\u7387\u89c6\u89d2\uff0c\u5e76\u642d\u51fa\u53ef\u89e3\u91ca\u7684\u6734\u7d20\u7ba1\u9053\u3002",
      output: "\u7aef\u5230\u7aef\u6d41\u7a0b\u56fe + 8 \u4e2a\u51b3\u7b56\u70b9\u6e05\u5355",
      acceptance: "\u80fd\u89e3\u91ca RAG-Sequence / Token \u4e0e\u79bb\u7ebf\u3001\u5728\u7ebf\u4e24\u6761\u6d41\u6c34\u7ebf\u3002"
    },
    d03: {
      hours: "2.5-3 \u5c0f\u65f6",
      objective: "\u638c\u63e1\u89e3\u6790\u548c\u5206\u5757\u5982\u4f55\u51b3\u5b9a\u540e\u7eed\u68c0\u7d22\u8d28\u91cf\uff0c\u5e76\u5efa\u7acb\u53ef\u8bc4\u4f30\u7684\u9009\u578b\u65b9\u6cd5\u3002",
      output: "\u89e3\u6790\u5668\u4e0e\u5206\u5757\u7b56\u7565\u77e9\u9635 + 5 \u4e2a\u5931\u8d25\u6837\u4f8b",
      acceptance: "\u80fd\u6839\u636e\u6587\u6863\u7ed3\u6784\u3001\u67e5\u8be2\u7c7b\u578b\u548c\u6307\u6807\u9009\u62e9 chunk \u7b56\u7565\u3002"
    },
    d04: {
      hours: "2.5-3 \u5c0f\u65f6",
      objective: "\u4ece\u5bf9\u6bd4\u5b66\u4e60\u3001\u53cc\u5854\u4e0e\u8d1f\u4f8b\u5de5\u7a0b\u7406\u89e3 Embedding\uff0c\u800c\u4e0d\u662f\u505c\u7559\u5728 API \u8c03\u7528\u3002",
      output: "InfoNCE \u63a8\u5bfc + Embedding \u9009\u578b\u8bb0\u5f55",
      acceptance: "\u80fd\u89e3\u91ca pooling\u3001\u5f52\u4e00\u5316\u3001\u96be\u8d1f\u4f8b\u4e0e\u9886\u57df\u5fae\u8c03\u3002"
    },
    d05: {
      hours: "2.5-3 \u5c0f\u65f6",
      objective: "\u5efa\u7acb ANN \u7684\u6570\u5b57\u611f\uff0c\u7406\u89e3 IVF\u3001PQ\u3001HNSW \u7684\u7cbe\u5ea6\u3001\u5ef6\u8fdf\u4e0e\u5185\u5b58\u53d6\u820d\u3002",
      output: "\u7d22\u5f15\u5bf9\u6bd4\u5b9e\u9a8c\u8868 + \u4e00\u9053\u5bb9\u91cf\u4f30\u7b97",
      acceptance: "\u80fd\u6309\u89c4\u6a21\u3001\u8fc7\u6ee4\u3001\u66f4\u65b0\u548c SLA \u9009\u62e9\u7d22\u5f15\u3002"
    },
    d06: {
      hours: "2.5-3 \u5c0f\u65f6",
      objective: "\u628a BM25\u3001\u7a20\u5bc6\u68c0\u7d22\u3001RRF \u4e0e Cross-Encoder \u4e32\u6210\u9ad8\u8d28\u91cf\u53ec\u56de\u94fe\u8def\u3002",
      output: "\u6df7\u5408\u68c0\u7d22\u6d88\u878d\u8868 + \u7ea7\u8054\u67b6\u6784\u56fe",
      acceptance: "\u80fd\u89e3\u91ca\u4e3a\u4ec0\u4e48\u878d\u5408\u7528\u6392\u540d\u3001\u4e3a\u4ec0\u4e48\u7cbe\u6392\u4e0d\u80fd\u76f4\u63a5\u641c\u5168\u5e93\u3002"
    },
    d07: {
      hours: "2.5-3 \u5c0f\u65f6",
      objective: "\u638c\u63e1\u67e5\u8be2\u6539\u5199\u3001HyDE\u3001\u8def\u7531\u4e0e Self-RAG \u7684\u89e6\u53d1\u6761\u4ef6\u548c\u6210\u672c\u8fb9\u754c\u3002",
      output: "\u67e5\u8be2\u7b56\u7565\u51b3\u7b56\u6811 + 10 \u6761\u6539\u5199\u6837\u4f8b",
      acceptance: "\u80fd\u533a\u5206\u4f55\u65f6\u6539\u5199\u3001\u4f55\u65f6\u591a\u67e5\u8be2\u3001\u4f55\u65f6\u6309\u9700\u68c0\u7d22\u3002"
    }
  };

  Object.assign(dayBriefs, {
    d08: {
      hours: "2.5-3 \u5c0f\u65f6",
      objective: "\u6bd4\u8f83 CRAG\u3001GraphRAG \u4e0e\u4e0a\u4e0b\u6587\u7ec4\u7ec7\u7b56\u7565\uff0c\u5b66\u4f1a\u6309\u95ee\u9898\u7ed3\u6784\u9009\u67b6\u6784\u3002",
      output: "\u9ad8\u7ea7 RAG \u67b6\u6784\u51b3\u7b56\u8bb0\u5f55 ADR",
      acceptance: "\u80fd\u8bf4\u660e\u6bcf\u79cd\u67b6\u6784\u89e3\u51b3\u7684\u5931\u8d25\u6a21\u5f0f\u4e0e\u65b0\u589e\u4ee3\u4ef7\u3002"
    },
    d09: {
      hours: "3 \u5c0f\u65f6",
      objective: "\u7528 LangGraph \u843d\u5730\u53ef\u89c2\u6d4b\u3001\u53ef\u9650\u6b21\u3001\u53ef\u56de\u9000\u7684 Agentic RAG\u3002",
      output: "\u53ef\u8fd0\u884c CRAG \u72b6\u6001\u56fe + trace \u6837\u4f8b",
      acceptance: "\u80fd\u753b\u51fa\u72b6\u6001\u3001\u6761\u4ef6\u8fb9\u3001\u5faa\u73af\u9884\u7b97\u4e0e\u5931\u8d25\u51fa\u53e3\u3002"
    },
    d10: {
      hours: "2.5-3 \u5c0f\u65f6",
      objective: "\u638c\u63e1\u591a\u6a21\u6001\u6587\u6863\u68c0\u7d22\u8def\u7ebf\uff0c\u5e76\u5efa\u7acb\u68c0\u7d22\u4e0e\u751f\u6210\u5206\u5c42\u6307\u6807\u3002",
      output: "\u591a\u6a21\u6001\u8def\u7ebf\u5bf9\u6bd4 + \u6307\u6807\u5b9a\u4e49\u8868",
      acceptance: "\u80fd\u89e3\u91ca ColPali / ColBERT \u601d\u60f3\u4e0e Recall\u3001MRR\u3001nDCG\u3001Faithfulness\u3002"
    },
    d11: {
      hours: "3 \u5c0f\u65f6",
      objective: "\u5efa\u7acb\u8bc4\u4f30\u6570\u636e\u3001\u7ebf\u4e0a\u89c2\u6d4b\u3001\u7d22\u5f15\u751f\u547d\u5468\u671f\u3001\u5b89\u5168\u4e0e\u6545\u969c\u5f52\u56e0\u95ed\u73af\u3002",
      output: "50 \u6761\u8bc4\u4f30\u96c6\u89c4\u683c + \u751f\u4ea7\u6545\u969c\u624b\u518c",
      acceptance: "\u80fd\u4ece trace \u5224\u65ad\u68c0\u7d22\u5931\u8d25\u8fd8\u662f\u751f\u6210\u5931\u8d25\uff0c\u5e76\u7ed9\u51fa\u56de\u5f52\u65b9\u6848\u3002"
    },
    d12: {
      hours: "2-2.5 \u5c0f\u65f6",
      objective: "\u96c6\u4e2d\u8bad\u7ec3 RAG \u57fa\u7840\u3001Embedding\u3001\u6df7\u5408\u68c0\u7d22\u3001\u5206\u5757\u4e0e\u6392\u969c\u8868\u8fbe\u3002",
      output: "5 \u9053 90 \u79d2\u5f55\u97f3\u6216\u9010\u5b57\u7a3f",
      acceptance: "\u6bcf\u9898\u90fd\u80fd\u6309\u7ed3\u8bba\u3001\u673a\u5236\u3001\u53d6\u820d\u3001\u573a\u666f\u56db\u6bb5\u4f5c\u7b54\u3002"
    },
    d13: {
      hours: "2.5-3 \u5c0f\u65f6",
      objective: "\u8bad\u7ec3\u8bc4\u4f30\u3001Agentic RAG\u3001\u591a\u8df3\u3001\u591a\u6a21\u6001\u4e0e\u4f01\u4e1a\u77e5\u8bc6\u5e93\u7cfb\u7edf\u8bbe\u8ba1\u3002",
      output: "6 \u9053\u6df1\u6316\u7b54\u6848 + \u4e00\u5f20\u7cfb\u7edf\u8bbe\u8ba1\u767d\u677f\u56fe",
      acceptance: "\u80fd\u4e3b\u52a8\u6f84\u6e05\u89c4\u6a21\u3001SLA\u3001ACL\u3001\u6210\u672c\u548c\u8bc4\u4f30\u53e3\u5f84\u3002"
    },
    d14: {
      hours: "3-4 \u5c0f\u65f6",
      objective: "\u7528\u4e00\u4e2a\u7aef\u5230\u7aef\u9879\u76ee\u3001\u6d88\u878d\u6570\u636e\u548c\u6a21\u62df\u7b54\u8fa9\u5b8c\u6210\u77e5\u8bc6\u6536\u675f\u3002",
      output: "\u9879\u76ee README + \u6d88\u878d\u8868 + 30 \u5206\u949f\u6a21\u62df\u9762\u8bd5",
      acceptance: "\u80fd\u72ec\u7acb\u89e3\u91ca\u3001\u5b9e\u73b0\u3001\u8bc4\u4f30\u3001\u6392\u969c\u5e76\u5b8c\u6210\u67b6\u6784\u9009\u578b\u3002"
    }
  });

  const memoryStore = {};
  let state = loadState();
  let units = [];
  let currentIndex = 0;
  let checkpoint;
  let timerId = null;
  let timerRemaining = FOCUS_SECONDS;

  document.addEventListener("DOMContentLoaded", init);

  function init() {
    const main = document.querySelector("main");
    const nav = document.querySelector("nav.toc");
    const hero = document.querySelector("header.hero");
    const sections = [...document.querySelectorAll("main > section")];

    if (!main || !nav || !hero || !sections.length) return;

    units = buildUnits(sections);
    currentIndex = Math.min(Math.max(Number(state.current) || 0, 0), units.length - 1);
    currentIndex = Math.min(currentIndex, unlockedIndex());
    checkpoint = createCheckpoint();

    buildSidebar(nav);
    buildToolbar(hero);
    buildDayBriefing(main);
    buildBrowseBanner(main);
    document.body.classList.add("study-ready");

    setMode(state.mode === "browse" ? "browse" : "focus", false);
  }

  function buildUnits(sections) {
    const result = [];

    sections.forEach(section => {
      const sectionId = section.id;
      const sectionTitle = cleanText(section.querySelector(":scope > h2"));
      const children = [...section.children];
      const h3s = children.filter(element => element.tagName === "H3");
      const groups = [];

      if (h3s.length) {
        h3s.forEach((heading, index) => {
          const start = children.indexOf(heading);
          const end = index + 1 < h3s.length ? children.indexOf(h3s[index + 1]) : children.length;
          groups.push({ title: cleanText(heading), elements: children.slice(start, end) });
        });
      } else {
        const details = children.filter(element => element.tagName === "DETAILS");
        if (details.length) {
          details.forEach(detail => {
            groups.push({
              title: cleanText(detail.querySelector("summary")),
              elements: [detail]
            });
          });
        } else {
          const content = children.filter(element => {
            return element.tagName !== "H2" && !element.classList.contains("lead");
          });
          groups.push({ title: sectionTitle, elements: content });
        }
      }

      groups.forEach((group, index) => {
        const id = sectionId + "-u" + (index + 1);
        group.elements.forEach(element => {
          element.classList.add("unit-fragment");
          element.dataset.unitId = id;
        });

        const textLength = group.elements.reduce((sum, element) => {
          return sum + element.textContent.trim().length;
        }, 0);

        result.push({
          id,
          section,
          sectionId,
          sectionTitle,
          title: group.title,
          elements: group.elements,
          minutes: Math.min(18, Math.max(8, Math.ceil(textLength / 360) + 4)),
          day: days.find(day => day.unitIds.includes(id))
        });
      });
    });

    return result;
  }

  function buildSidebar(nav) {
    nav.innerHTML = [
      '<div class="study-brand">',
      '  <span class="study-brand-mark" aria-hidden="true">R</span>',
      '  <div><strong>RAG Learning Lab</strong><span>14 \u5929 / \u9879\u76ee / \u9762\u8bd5</span></div>',
      '</div>',
      '<div class="study-sidebar-progress">',
      '  <div class="study-progress-copy"><span>\u603b\u8fdb\u5ea6</span><strong id="study-progress-percent">0%</strong></div>',
      '  <div class="study-progress-track" aria-hidden="true"><div class="study-progress-fill" id="study-progress-fill"></div></div>',
      '  <p class="study-today">\u5f53\u524d Day <strong id="study-today-count">0 / 0</strong> \u4e2a\u5c0f\u8282</p>',
      '</div>',
      '<p class="study-phase-heading">14 \u5929\u5b66\u4e60\u8ba1\u5212</p>',
      '<div class="study-phase-list" id="study-phase-list"></div>',
      '<div class="study-sidebar-actions">',
      '  <a class="study-sidebar-button" href="14-day-learning-plan.md">\u603b\u7ef4\u62a4\u6587\u6863</a>',
      '  <button class="study-sidebar-button" id="study-mode-toggle" type="button">\u5207\u6362\u5230\u67e5\u9605\u6a21\u5f0f</button>',
      '  <button class="study-sidebar-button is-subtle" id="study-reset" type="button">\u91cd\u7f6e\u5b66\u4e60\u8fdb\u5ea6</button>',
      '</div>'
    ].join("");

    const dayList = nav.querySelector("#study-phase-list");
    days.forEach((day, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "study-phase";
      button.dataset.dayId = day.id;
      button.innerHTML = [
        '<span class="study-phase-no">' + String(index + 1).padStart(2, "0") + '</span>',
        '<span class="study-phase-label">' + day.label + '</span>',
        '<span class="study-phase-count"></span>'
      ].join("");
      button.addEventListener("click", () => navigateToDay(day));
      dayList.appendChild(button);
    });

    nav.querySelector("#study-mode-toggle").addEventListener("click", () => {
      setMode(state.mode === "focus" ? "browse" : "focus");
    });

    nav.querySelector("#study-reset").addEventListener("click", () => {
      if (!window.confirm("\u91cd\u7f6e\u540e\u4f1a\u6e05\u9664\u5b8c\u6210\u8bb0\u5f55\u3001\u81ea\u8bc4\u548c\u5b66\u4e60\u7b14\u8bb0\u3002\u786e\u5b9a\u7ee7\u7eed\u5417\uff1f")) return;
      state = defaultState();
      currentIndex = 0;
      saveState();
      setMode("focus", false);
    });
  }

  function buildToolbar(hero) {
    const toolbar = document.createElement("div");
    toolbar.className = "study-toolbar";
    toolbar.innerHTML = [
      '<div>',
      '  <p class="study-toolbar-kicker" id="study-phase-name">\u4eca\u65e5\u5b66\u4e60</p>',
      '  <h2 id="study-unit-title"></h2>',
      '  <p class="study-unit-meta">',
      '    <span id="study-unit-position"></span>',
      '    <span id="study-unit-time"></span>',
      '  </p>',
      '</div>',
      '<div class="study-timer" aria-label="\u4e13\u6ce8\u8ba1\u65f6\u5668">',
      '  <span class="study-timer-value" id="study-timer-value">25:00</span>',
      '  <button class="study-timer-button" id="study-timer-button" type="button">\u5f00\u59cb</button>',
      '</div>'
    ].join("");

    hero.insertAdjacentElement("afterend", toolbar);
    toolbar.querySelector("#study-timer-button").addEventListener("click", toggleTimer);
  }

  function buildDayBriefing(main) {
    const briefing = document.createElement("div");
    briefing.className = "study-day-briefing";
    briefing.innerHTML = [
      '<div class="study-day-heading">',
      '  <p class="study-day-kicker" id="study-day-kicker"></p>',
      '  <h2 id="study-day-title"></h2>',
      '  <p id="study-day-objective"></p>',
      '</div>',
      '<div class="study-day-contract">',
      '  <p><span>\u4eca\u65e5\u4ea7\u51fa</span><strong id="study-day-output"></strong></p>',
      '  <p><span>\u9a8c\u6536\u53e3\u5f84</span><strong id="study-day-acceptance"></strong></p>',
      '  <a id="study-day-document" href="days/day-01/index.html">\u8fdb\u5165 Day 01 \u4ea4\u4e92\u5b66\u4e60</a>',
      '</div>'
    ].join("");
    main.querySelector(".study-toolbar").insertAdjacentElement("afterend", briefing);
  }


  function buildBrowseBanner(main) {
    const banner = document.createElement("p");
    banner.className = "study-browse-banner";
    banner.textContent = "\u67e5\u9605\u6a21\u5f0f\uff1a\u5df2\u5c55\u5f00\u5168\u90e8\u7ae0\u8282\u3002\u5b66\u4e60\u8fdb\u5ea6\u4e0d\u4f1a\u56e0\u6d4f\u89c8\u800c\u6539\u53d8\u3002";
    main.querySelector(".study-toolbar").insertAdjacentElement("afterend", banner);
  }

  function createCheckpoint() {
    const panel = document.createElement("aside");
    panel.className = "study-checkpoint";
    panel.setAttribute("aria-labelledby", "study-checkpoint-title");
    panel.innerHTML = [
      '<p class="study-checkpoint-label">Recall checkpoint</p>',
      '<h3 id="study-checkpoint-title">\u5408\u4e0a\u6b63\u6587\uff0c\u5148\u8bb2\u4e00\u904d</h3>',
      '<p class="study-recall-prompt" id="study-recall-prompt"></p>',
      '<label class="study-note-label" for="study-notes">\u4f60\u7684\u590d\u8ff0\u4e0e\u7591\u95ee</label>',
      '<textarea class="study-notes" id="study-notes" placeholder="\u7528\u81ea\u5df1\u7684\u8bdd\u5199 3 \u5230 5 \u53e5\u3002\u4e0d\u8981\u6458\u6284\u539f\u6587\u3002"></textarea>',
      '<p class="study-rating-label">\u73b0\u5728\u7684\u638c\u63e1\u7a0b\u5ea6</p>',
      '<div class="study-rating" role="group" aria-label="\u638c\u63e1\u7a0b\u5ea6">',
      '  <button type="button" data-rating="fuzzy">\u8fd8\u6a21\u7cca</button>',
      '  <button type="button" data-rating="understand">\u57fa\u672c\u7406\u89e3</button>',
      '  <button type="button" data-rating="explain">\u80fd\u8bb2\u6e05\u695a</button>',
      '</div>',
      '<p class="study-interview-cue"><strong>\u9762\u8bd5\u8868\u8fbe\uff1a</strong>\u5148\u7ed9\u7ed3\u8bba\uff0c\u518d\u8bb2\u673a\u5236\uff0c\u63a5\u7740\u8bf4\u53d6\u820d\uff0c\u6700\u540e\u843d\u5230\u9879\u76ee\u573a\u666f\u3002</p>',
      '<div class="study-checkpoint-actions">',
      '  <button class="study-action" id="study-previous" type="button">\u4e0a\u4e00\u5c0f\u8282</button>',
      '  <button class="study-action is-primary" id="study-complete" type="button">\u5b8c\u6210\u5e76\u8fdb\u5165\u4e0b\u4e00\u8282</button>',
      '</div>'
    ].join("");

    panel.querySelector("#study-previous").addEventListener("click", () => navigate(currentIndex - 1));
    panel.querySelector("#study-complete").addEventListener("click", completeCurrent);
    panel.querySelector("#study-notes").addEventListener("input", event => {
      state.notes[units[currentIndex].id] = event.target.value;
      saveState();
    });
    panel.querySelectorAll("[data-rating]").forEach(button => {
      button.addEventListener("click", () => {
        state.ratings[units[currentIndex].id] = button.dataset.rating;
        saveState();
        renderCheckpoint();
      });
    });

    return panel;
  }

  function render() {
    if (!units.length) return;

    const unit = units[currentIndex];
    const isBrowse = state.mode === "browse";

    document.querySelectorAll("main > section").forEach(section => {
      section.hidden = !isBrowse && section !== unit.section;
    });

    units.forEach((candidate, index) => {
      candidate.elements.forEach(element => {
        element.hidden = !isBrowse && index !== currentIndex;
      });
    });

    if (!isBrowse) {
      unit.section.appendChild(checkpoint);
      renderToolbar();
      renderDayBriefing();
      renderCheckpoint();
    }

    renderSidebar();
    state.current = currentIndex;
    saveState();
  }

  function renderToolbar() {
    const unit = units[currentIndex];
    const dayUnits = units.filter(candidate => candidate.day === unit.day);
    const dayPosition = dayUnits.findIndex(candidate => candidate.id === unit.id) + 1;
    setText("study-phase-name", "DAY " + dayNumber(unit.day) + " / " + unit.day.label);
    setText("study-unit-title", unit.title);
    setText("study-unit-position", "\u672c\u65e5 " + dayPosition + " / " + dayUnits.length);
    setText("study-unit-time", "\u9884\u8ba1 " + unit.minutes + " \u5206\u949f");
  }

  function renderDayBriefing() {
    const day = units[currentIndex].day;
    const brief = dayBriefs[day.id];
    const number = dayNumber(day);

    setText("study-day-kicker", "DAY " + number + " / 14 \u00b7 " + brief.hours);
    setText("study-day-title", day.label);
    setText("study-day-objective", brief.objective);
    setText("study-day-output", brief.output);
    setText("study-day-acceptance", brief.acceptance);

    const documentLink = document.querySelector("#study-day-document");
    const hasSession = ["d01", "d02"].includes(day.id);
    documentLink.href = hasSession
      ? "days/day-" + number + "/index.html"
      : "14-day-learning-plan.md";
    documentLink.textContent = hasSession
      ? "\u8fdb\u5165 Day " + number + " \u4ea4\u4e92\u5b66\u4e60"
      : "\u67e5\u770b 14 \u5929\u603b\u8ba1\u5212";
  }

  function renderCheckpoint() {
    const unit = units[currentIndex];
    const rating = state.ratings[unit.id] || "";
    const completeButton = checkpoint.querySelector("#study-complete");
    const isCompleted = Boolean(state.completedAt[unit.id]);

    setText("study-recall-prompt", recallPrompt(unit));
    checkpoint.querySelector("#study-notes").value = state.notes[unit.id] || "";
    checkpoint.querySelectorAll("[data-rating]").forEach(button => {
      button.classList.toggle("is-selected", button.dataset.rating === rating);
    });
    checkpoint.querySelector("#study-previous").disabled = currentIndex === 0;

    completeButton.disabled = !rating;
    if (isCompleted) {
      completeButton.textContent = currentIndex === units.length - 1 ? "\u672c\u8bfe\u7a0b\u5df2\u5b8c\u6210" : "\u8fdb\u5165\u4e0b\u4e00\u5c0f\u8282";
    } else {
      completeButton.textContent = rating ? "\u5b8c\u6210\u5e76\u8fdb\u5165\u4e0b\u4e00\u8282" : "\u81ea\u8bc4\u540e\u7ee7\u7eed";
    }
  }

  function renderSidebar() {
    const completedCount = Object.keys(state.completedAt).filter(id => {
      return units.some(unit => unit.id === id);
    }).length;
    const percent = Math.round((completedCount / units.length) * 100);
    const unlocked = unlockedIndex();
    const currentDay = units[currentIndex].day;
    const currentDayUnits = units.filter(unit => unit.day === currentDay);
    const currentDayCompleted = currentDayUnits.filter(unit => state.completedAt[unit.id]).length;

    setText("study-progress-percent", percent + "%");
    setText("study-today-count", currentDayCompleted + " / " + currentDayUnits.length);
    const fill = document.querySelector("#study-progress-fill");
    if (fill) fill.style.width = percent + "%";

    document.querySelectorAll(".study-phase").forEach(button => {
      const day = days.find(candidate => candidate.id === button.dataset.dayId);
      const dayUnits = units.filter(unit => unit.day === day);
      const completed = dayUnits.filter(unit => state.completedAt[unit.id]).length;
      const firstIndex = units.findIndex(unit => unit.day === day);
      const count = button.querySelector(".study-phase-count");

      count.textContent = completed + "/" + dayUnits.length;
      button.classList.toggle("is-active", day === currentDay);
      button.disabled = state.mode === "focus" && firstIndex > unlocked;
    });

    const modeButton = document.querySelector("#study-mode-toggle");
    if (modeButton) {
      modeButton.textContent = state.mode === "focus" ? "\u5207\u6362\u5230\u67e5\u9605\u6a21\u5f0f" : "\u8fd4\u56de\u5b66\u4e60\u6a21\u5f0f";
    }
  }

  function completeCurrent() {
    const unit = units[currentIndex];
    if (!state.ratings[unit.id]) return;

    if (!state.completedAt[unit.id]) state.completedAt[unit.id] = todayKey();
    saveState();

    if (currentIndex < units.length - 1) navigate(currentIndex + 1);
    else render();
  }

  function navigate(index) {
    if (index < 0 || index >= units.length) return;
    if (state.mode === "focus" && index > unlockedIndex()) return;

    currentIndex = index;
    render();
    window.scrollTo({
      top: document.querySelector(".study-toolbar").offsetTop,
      behavior: "smooth"
    });
  }

  function navigateToDay(day) {
    const dayIndexes = units
      .map((unit, index) => ({ unit, index }))
      .filter(item => item.unit.day === day);
    if (!dayIndexes.length) return;

    if (state.mode === "browse") {
      dayIndexes[0].unit.elements[0].scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }

    const target = dayIndexes.find(item => !state.completedAt[item.unit.id]) || dayIndexes[0];
    navigate(Math.min(target.index, unlockedIndex()));
  }

  function setMode(mode, persist = true) {
    state.mode = mode;
    document.body.classList.toggle("is-browse-mode", mode === "browse");
    if (persist) saveState();
    render();
  }

  function unlockedIndex() {
    const firstIncomplete = units.findIndex(unit => !state.completedAt[unit.id]);
    return firstIncomplete === -1 ? units.length - 1 : firstIncomplete;
  }

  function recallPrompt(unit) {
    const template = promptTemplates[unit.sectionId] || (title => {
      return "\u7528\u81ea\u5df1\u7684\u8bdd\u89e3\u91ca\u201c" + title + "\u201d\uff0c\u5e76\u8bf4\u51fa\u5b83\u7684\u4ef7\u503c\u548c\u8fb9\u754c\u3002";
    });
    return template(unit.title);
  }

  function dayNumber(day) {
    return String(days.indexOf(day) + 1).padStart(2, "0");
  }

  function toggleTimer() {
    const button = document.querySelector("#study-timer-button");
    if (timerId) {
      window.clearInterval(timerId);
      timerId = null;
      button.textContent = "\u7ee7\u7eed";
      button.classList.remove("is-running");
      return;
    }

    if (timerRemaining <= 0) timerRemaining = FOCUS_SECONDS;
    button.textContent = "\u6682\u505c";
    button.classList.add("is-running");
    timerId = window.setInterval(() => {
      timerRemaining -= 1;
      renderTimer();
      if (timerRemaining <= 0) {
        window.clearInterval(timerId);
        timerId = null;
        button.textContent = "\u91cd\u5f00";
        button.classList.remove("is-running");
        document.title = "\u672c\u8f6e\u5b66\u4e60\u5b8c\u6210 \u00b7 RAG \u6df1\u5ea6\u638c\u63e1\u624b\u518c";
      }
    }, 1000);
  }

  function renderTimer() {
    const minutes = Math.floor(timerRemaining / 60);
    const seconds = timerRemaining % 60;
    setText(
      "study-timer-value",
      String(minutes).padStart(2, "0") + ":" + String(seconds).padStart(2, "0")
    );
  }

  function cleanText(element) {
    return element ? element.textContent.replace(/\s+/g, " ").trim() : "";
  }

  function todayKey() {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, "0");
    const day = String(now.getDate()).padStart(2, "0");
    return year + "-" + month + "-" + day;
  }

  function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
  }

  function defaultState() {
    return { current: 0, mode: "focus", completedAt: {}, ratings: {}, notes: {} };
  }

  function loadState() {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      const merged = saved ? { ...defaultState(), ...JSON.parse(saved) } : defaultState();
      merged.completedAt = merged.completedAt || {};
      merged.ratings = merged.ratings || {};
      merged.notes = merged.notes || {};
      return merged;
    } catch (error) {
      return { ...defaultState(), ...memoryStore };
    }
  }

  function saveState() {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (error) {
      Object.assign(memoryStore, state);
    }
  }
})();
