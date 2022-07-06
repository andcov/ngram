use std::collections::{HashMap, HashSet};
use std::fs;

fn main() {
    let text = fs::read_to_string("thor.txt").unwrap();
    let uni = Unigram::construt(&text);
    let bi = Bigram::construt(&text);

    println!("unigram: {}", uni.random_sentence());
    println!("bigram: {}", bi.random_sentence());

    println!();

    println!("uni pp: {}", uni.perplexity(&text));
    println!("bi pp: {}", bi.perplexity(&text));
}

pub const START_TOKEN: &str = "<s>";
pub const END_TOKEN: &str = "</s>";

struct Unigram {
    pub corpus: String,
    corpus_len: usize,
    pub vocab: HashSet<String>,
    word_counts: HashMap<String, usize>,
    pub probs: Vec<(String, f64)>,
}

impl Unigram {
    pub fn construt(text: &str) -> Self {
        let mut unigram = Unigram {
            corpus: clean_text(text.to_string()),
            corpus_len: 0,
            vocab: HashSet::new(),
            word_counts: HashMap::new(),
            probs: vec![],
        };

        unigram.vocab.insert(START_TOKEN.to_string());
        unigram.word_counts.insert(START_TOKEN.to_string(), 0);
        unigram.vocab.insert(END_TOKEN.to_string());
        unigram.word_counts.insert(END_TOKEN.to_string(), 0);

        for l in unigram.corpus.lines() {
            if l == "" {
                continue;
            }
            *unigram.word_counts.get_mut(START_TOKEN).unwrap() += 1;
            *unigram.word_counts.get_mut(END_TOKEN).unwrap() += 1;
            unigram.corpus_len += 2;
            for word in l.split(" ") {
                unigram.vocab.insert(word.to_string());
                let cnt = unigram.word_counts.entry(word.to_string()).or_insert(0);
                *cnt += 1;
                unigram.corpus_len += 1;
            }
        }

        let mut probs = vec![];
        for w in unigram.vocab.iter() {
            probs.push((w.to_string(), unigram.prob(w)));
        }
        probs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().reverse());
        unigram.probs = probs;

        unigram
    }

    pub fn prob(&self, word: &str) -> f64 {
        (*self.word_counts.get(word).unwrap_or_else(|| &0) as f64 / self.corpus_len as f64)
            .clamp(0.0, 1.0)
    }

    pub fn random_sentence(&self) -> String {
        let mut res = START_TOKEN.to_string();
        for _ in 0..100 {
            let val = rand::random::<f64>();
            let mut sum = 0.0;

            for (w, p) in self.probs.iter() {
                if sum + p > val {
                    if w == START_TOKEN {
                        continue;
                    }
                    res.push_str(" ");
                    res.push_str(w);
                    if w == END_TOKEN {
                        return res;
                    }
                    break;
                }
                sum += p;
            }
        }

        res.push_str(" ");
        res.push_str(END_TOKEN);
        res
    }

    pub fn perplexity(&self, text: &str) -> f64 {
        let text = clean_text(text.to_string());
        let mut cnt = 0;
        let mut sum = 0.0;

        for l in text.lines() {
            if l == "" {
                continue;
            }
            let l = format!("{} {} {}", START_TOKEN, l, END_TOKEN);
            for w in l.split(" ") {
                cnt += 1;
                sum += self.prob(w).ln();
            }
        }

        ((-1.0 / cnt as f64) * sum).exp()
    }
}

struct Bigram {
    pub corpus: String,
    pub vocab: HashSet<String>,
    bigram_counts: HashMap<String, HashMap<String, usize>>,
    word_counts: HashMap<String, usize>,
    pub probs: HashMap<String, Vec<(String, f64)>>,
}

impl Bigram {
    pub fn construt(text: &str) -> Self {
        let mut bigram = Bigram {
            corpus: clean_text(text.to_string()),
            vocab: HashSet::new(),
            bigram_counts: HashMap::new(),
            word_counts: HashMap::new(),
            probs: HashMap::new(),
        };

        bigram.vocab.insert(START_TOKEN.to_string());
        bigram.vocab.insert(END_TOKEN.to_string());

        for l in bigram.corpus.lines() {
            if l == "" {
                continue;
            }
            let l = format!("{} {}", l, END_TOKEN);

            let mut prev_token = START_TOKEN.to_string();
            for curr_token in l.split(" ") {
                bigram.vocab.insert(curr_token.to_string());
                *bigram
                    .bigram_counts
                    .entry(prev_token.to_string())
                    .or_insert(HashMap::new())
                    .entry(curr_token.to_string())
                    .or_insert(0) += 1;
                *bigram.word_counts.entry(prev_token).or_insert(0) += 1;

                prev_token = curr_token.to_string();
            }
        }

        let mut probs = HashMap::new();
        for w1 in bigram.vocab.iter() {
            let mut curr_probs = vec![];
            for w2 in bigram.vocab.iter() {
                curr_probs.push((w2.to_string(), bigram.prob(w1, w2)));
            }
            curr_probs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().reverse());
            probs.insert(w1.to_string(), curr_probs);
        }
        bigram.probs = probs;

        bigram
    }

    pub fn prob(&self, prev_word: &str, curr_word: &str) -> f64 {
        match self.bigram_counts.get(prev_word) {
            Some(hs) => (*hs.get(curr_word).unwrap_or_else(|| &0) as f64
                / *self
                    .word_counts
                    .get(prev_word)
                    .unwrap_or_else(|| &10000000000) as f64)
                .clamp(0.0, 1.0),
            None => 0.0,
        }
    }

    pub fn random_sentence(&self) -> String {
        let mut res = START_TOKEN.to_string();
        let mut prev_token = START_TOKEN.to_string();
        for _ in 0..100 {
            let val = rand::random::<f64>();
            let mut sum = 0.0;

            for (w, p) in self.probs.get(&prev_token).unwrap().iter() {
                if sum + p > val {
                    res.push_str(" ");
                    res.push_str(w);
                    if w == END_TOKEN {
                        return res;
                    }
                    prev_token = w.to_string();
                    break;
                }
                sum += p;
            }
        }

        res.push_str(" ");
        res.push_str(END_TOKEN);
        res
    }

    pub fn perplexity(&self, text: &str) -> f64 {
        let text = clean_text(text.to_string());
        let mut cnt = 0;
        let mut sum = 0.0;

        for l in text.lines() {
            if l == "" {
                continue;
            }
            let l = format!("{} {}", l, END_TOKEN);
            let mut prev_token = START_TOKEN.to_string();
            for curr_token in l.split(" ") {
                cnt += 1;
                sum += self.prob(&prev_token, curr_token).ln();
                prev_token = curr_token.to_string();
            }
        }

        ((-1.0 / cnt as f64) * sum).exp()
    }
}

pub fn clean_text(text: String) -> String {
    let mut text = text.replace(".", "");
    text = text.replace(",", "");
    text = text.replace("\"", "");
    text = text.replace("'", "");
    text = text.replace("?", "");
    text = text.replace("!", "");
    text = text.to_lowercase();

    text
}
