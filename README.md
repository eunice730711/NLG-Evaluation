# NLG-Evaluation
Natural Language Generation Evaluation by BLEU and METEOR

	Usage:
	python evaluation.py ./data/hypothesis.txt ./data/references1.txt ./data/references2.txt ./data/references3.txt

	Output:
	BLEU SCORE: 32.43
	METEOR SCORE: 56.88


references:
1. https://github.com/neubig/nn4nlp-code/blob/master/08-condlm/bleu.py
2. K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "BLEU: a method for automatic evaluation of machine translation," in Proceedings of the 40th annual meeting on association for computational linguistics, 2002, pp. 311-318: Association for Computational Linguistics.
3. S. Banerjee and A. Lavie, "METEOR: An automatic metric for MT evaluation with improved correlation with human judgments," in Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, 2005, pp. 65-72.