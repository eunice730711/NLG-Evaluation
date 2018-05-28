import math
from collections import Counter
import numpy
import sys

def bleu_stats(hypothesis, reference1, reference2, reference3):
  stats = []
  hypo_len = len(hypothesis)
  stats.append(hypo_len)
  # Find Closest Reference Length
  best_match_len = len(reference1)
  if abs(hypo_len-len(reference2))< abs(hypo_len-best_match_len):
    best_match_len = len(reference2)
  if abs(hypo_len-len(reference3))< abs(hypo_len-best_match_len):
    best_match_len = len(reference3)
  stats.append(best_match_len)

  # Summation of Unigram, Bigram, Trigram, 4-gram
  for n in range(1,5):
    s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)+1-n)])
    r1_ngrams = Counter([tuple(reference1[i:i+n]) for i in range(len(reference1)+1-n)])
    r2_ngrams = Counter([tuple(reference2[i:i+n]) for i in range(len(reference2)+1-n)])
    r3_ngrams = Counter([tuple(reference3[i:i+n]) for i in range(len(reference3)+1-n)])
    count_clip = max([sum(((s_ngrams & r1_ngrams) | (s_ngrams & r2_ngrams) | (s_ngrams & r3_ngrams)).values()), 0])
    stats.append(count_clip)
    stats.append(max([len(hypothesis)+1-n, 0]))
  return stats

# Compute BLEU from collected statistics obtained by call(s) to bleu_stats
def bleu(stats):
  if len(list(filter(lambda x: x==0, stats))) > 0:
    return 0
  # Compute brevity penalty
  (c, r) = stats[:2]
  log_bleu_prec = sum([math.log(float(x)/y) for x,y in zip(stats[2::2],stats[3::2])]) / 4.
  return math.exp(min([0, 1-float(r)/c]) + log_bleu_prec)*100


# METEOR SCORE(only exact mode)
def meteor_stats(hypothesis, reference1, reference2, reference3):
  stats = []
  hypo_len = len(hypothesis)
  # Find Closest Reference Length
  best_match_len = len(reference1)
  if abs(hypo_len-len(reference2))< abs(hypo_len-best_match_len):
    best_match_len = len(reference2)
  if abs(hypo_len-len(reference3))< abs(hypo_len-best_match_len):
    best_match_len = len(reference3)

  for n in range(1,3):
    s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)+1-n)])
    r1_ngrams = Counter([tuple(reference1[i:i+n]) for i in range(len(reference1)+1-n)])
    r2_ngrams = Counter([tuple(reference2[i:i+n]) for i in range(len(reference2)+1-n)])
    r3_ngrams = Counter([tuple(reference3[i:i+n]) for i in range(len(reference3)+1-n)])
    stats.append(max([sum(((s_ngrams & r1_ngrams) | (s_ngrams & r2_ngrams) | (s_ngrams & r3_ngrams)).values()), 0]))
  stats.append(max([len(hypothesis)+1-1, 0])) # For Unigram Precision
  stats.append(max([best_match_len+1-1, 0])) # For Unigram Recall
  return stats

def meteor(m_stats):
  unigram_prec = float(m_stats[0]/m_stats[2])
  unigram_recall = float(m_stats[0]/m_stats[3])
  chunk_num = m_stats[0]-m_stats[1]
  fmean = 10*unigram_recall*unigram_prec/(9*unigram_prec+unigram_recall)
  score = fmean * (1-0.5*(chunk_num/m_stats[0])*(chunk_num/m_stats[0])*(chunk_num/m_stats[0]))
  return score*100

if __name__=='__main__':
  b_stats = numpy.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
  m_stats = numpy.array([0., 0., 0., 0.])
  for hyp, ref1, ref2, ref3 in zip(open(sys.argv[1], 'r'), open(sys.argv[2], 'r'), open(sys.argv[3], 'r'), open(sys.argv[4], 'r')):
    hyp, ref1, ref2, ref3 = (hyp.strip().split(), ref1.strip().split(), ref2.strip().split(), ref3.strip().split())
    b_stats += numpy.array(bleu_stats(hyp, ref1, ref2, ref3))
    m_stats += numpy.array(meteor_stats(hyp, ref1, ref2, ref3))

  print("BLEU SCORE: %.2f" % (bleu(b_stats)))
  print("METEOR SCORE: %.2f" % meteor(m_stats))
