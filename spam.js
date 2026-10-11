// Spam classifier that runs fully in the browser.
// It repeats the same cleaning + TF-IDF steps used in training (preprocess.py / train_export.py).
(function (root) {
  const URL_RE = /https?:\/\/\S+|www\.\S+/g;
  const EMAIL_RE = /\S+@\S+\.\S+/g;
  const TAG_RE = /<[^>]+>/g;
  const STYLE_RE = /<(script|style).*?>.*?<\/\1>/gis;
  const SUBJECT_RE = /^subject:\s*(.*)$/im;

  const NAMED_ENTITIES = {
    amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ",
  };

  function decodeEntities(text) {
    return text
      .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
      .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
      .replace(/&([a-z]+);/gi, (m, name) => NAMED_ENTITIES[name.toLowerCase()] ?? m);
  }

  function stripHtml(text) {
    return decodeEntities(text.replace(STYLE_RE, " ").replace(TAG_RE, " "));
  }

  function prepareEmail(raw) {
    raw = String(raw || "");

    let subject = "";
    const match = SUBJECT_RE.exec(raw.slice(0, 1000));
    if (match) {
      subject = match[1];
      raw = raw.slice(0, match.index) + raw.slice(match.index + match[0].length);
    }

    let text = subject + " " + subject + " " + raw;
    text = stripHtml(text).toLowerCase();
    text = text.replace(URL_RE, " urltoken ");
    text = text.replace(EMAIL_RE, " emailtoken ");
    text = text.replace(/\d+/g, " numtoken ");
    text = text.replace(/[^a-z\s]/g, " ");
    text = text.replace(/\s+/g, " ").trim();
    return text;
  }

  const FRIENDLY = {
    urltoken: "[web link]",
    emailtoken: "[email address]",
    numtoken: "[number]",
  };

  function friendly(term) {
    return term
      .split(" ")
      .map((w) => FRIENDLY[w] || w)
      .join(" ");
  }

  function create(model) {
    const vocab = new Map();
    model.terms.forEach((t, i) => vocab.set(t, i));
    const stop = new Set(model.stop_words);

    function predict(subject, body) {
      const subj = String(subject || "").trim();
      const text = String(body || "").trim();
      const raw = subj ? "Subject: " + subj + "\n" + text : text;
      const cleaned = prepareEmail(raw);
      if (!cleaned) return null;

      // Facts about the cleaning step, for the explanation view
      const noUrls = raw.replace(URL_RE, " ");
      const replaced = {
        links: (raw.match(URL_RE) || []).length,
        emails: (noUrls.match(EMAIL_RE) || []).length,
        numbers: (noUrls.replace(EMAIL_RE, " ").match(/\d+/g) || []).length,
        htmlTags: (raw.match(TAG_RE) || []).length,
      };

      // Same as scikit-learn: drop stop words, then build unigrams and bigrams
      const tokens = (cleaned.match(/\b\w\w+\b/g) || []).filter((t) => !stop.has(t));
      const counts = new Map();
      const add = (term) => {
        if (vocab.has(term)) counts.set(term, (counts.get(term) || 0) + 1);
      };
      for (let i = 0; i < tokens.length; i++) {
        add(tokens[i]);
        if (i + 1 < tokens.length) add(tokens[i] + " " + tokens[i + 1]);
      }

      // TF-IDF with sublinear tf and L2 normalisation
      const values = [];
      let norm = 0;
      for (const [term, count] of counts) {
        const idx = vocab.get(term);
        const x = (1 + Math.log(count)) * model.idf[idx];
        values.push({ term, idx, x });
        norm += x * x;
      }
      norm = Math.sqrt(norm);

      let score = model.bias;
      const contributions = [];
      if (norm > 0) {
        for (const v of values) {
          const c = (v.x / norm) * model.weights[v.idx];
          score += c;
          contributions.push({ term: friendly(v.term), value: c });
        }
      }

      const probability = 1 / (1 + Math.exp(-score));
      contributions.sort((a, b) => b.value - a.value);

      return {
        spamProbability: probability,
        label: probability >= 0.5 ? "spam" : "ham",
        spamWords: contributions.filter((c) => c.value > 0).slice(0, 5),
        hamWords: contributions.filter((c) => c.value < 0).slice(-5).reverse(),
        featuresFound: contributions.length,
        contributions,
        cleaned,
        replaced,
        tokens: tokens.length,
        vocabSize: model.terms.length,
        bias: model.bias,
        score,
      };
    }

    return { predict };
  }

  const api = { create, prepareEmail };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SpamDetector = api;
})(typeof window !== "undefined" ? window : globalThis);