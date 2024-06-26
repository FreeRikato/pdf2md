#

# Corrective Retrieval Augmented Generation

Shi-Qi Yan1*, Jia-Chen Gu2*, Yun Zhu3, Zhen-Hua Ling1

1National Engineering Research Center of Speech and Language Information Processing, University of Science and Technology of China, Hefei, China

2Department of Computer Science, University of California, Los Angeles3Google Research

yansiki@mail.ustc.edu.cn, gujc@ucla.edu, yunzhu@google.com, zhling@ustc.edu.cn

# Abstract

Large language models (LLMs) inevitably exhibit hallucinations since the accuracy of generated texts cannot be secured solely by the parametric knowledge they encapsulate. Although retrieval-augmented generation (RAG) is a practicable complement to LLMs, it relies heavily on the relevance of retrieved documents, raising concerns about how the model behaves if retrieval goes wrong. To this end, we propose the Corrective Retrieval Augmented Generation (CRAG) to improve the robustness of generation. Specifically, a lightweight retrieval evaluator is designed to assess the overall quality of retrieved documents for a query, returning a confidence degree based on which different knowledge retrieval actions can be triggered. Since retrieval from static and limited corpora can only return sub-optimal documents, large-scale web searches are utilized as an extension for augmenting the retrieval results. Besides, a decompose-then-recompose algorithm is designed for retrieved documents to selectively focus on key information and filter out irrelevant information in them. CRAG is plug-and-play and can be seamlessly coupled with various RAG-based approaches. Experiments on four datasets covering short- and long-form generation tasks show that CRAG can significantly improve the performance of RAG-based approaches.1

# Introduction

Large language models (LLMs) have attracted increasing attention and exhibited impressive abilities to understand instructions and generate fluent language texts (Brown et al., 2020; Ouyang et al., 2022; Touvron et al., 2023a). Nevertheless, LLMs inevitably manifest hallucinations (Ji et al., 2023) due to their struggle with factual errors (Mallen et al., 2023; Min et al., 2023) and inability to secure the accuracy of generated texts solely by

1* Equal contribution.

The code is available at github.com/HuskyInSalt/CRAG
---
# Impeding the models from acquiring accurate knowledge and potentially misleading them, resulting in issues such as hallucinations (Zhang et al., 2023b). However, most conventional RAG approaches indiscriminately incorporate the retrieved documents, regardless of whether these documents are relevant or not (Rony et al., 2022). Furthermore, current methods mostly treat complete documents as reference knowledge both during retrieval and utilization. But a considerable portion of the text within these retrieved documents is often non-essential for generation, which should not have been equally referred to and involved in RAG.

On account of the above issues, this paper particularly studies the scenarios where the retriever returns inaccurate results. A method named Corrective Retrieval-Augmented Generation (CRAG) is proposed to self-correct the results of retriever and improve the utilization of documents for augmenting generation. A lightweight retrieval evaluator is designed to assess the overall quality of retrieved documents for a query. This serves as a crucial component in RAG, contributing to informative generation by reviewing and evaluating the relevance and reliability of the retrieved documents. A confidence degree is quantified based on which different knowledge retrieval actions of {Correct, Incorrect, Ambiguous} can be triggered. For the latter two actions, large-scale web searches (Piktus et al., 2021; Komeili et al., 2022) are integrated as a strategic extension, since retrieval from static and limited corpora can only return sub-optimal documents in terms of scope and diversity. This augmentation is implemented to broaden the spectrum of retrieved information, harnessing the expansive and dynamic nature of the web to complement and enrich the initially obtained documents. Furthermore, to eliminate redundant contexts contained in retrieved documents that are unhelpful for RAG, a decompose-then-recompose algorithm is meticulously crafted throughout the retrieval and utilization process. This algorithm ensures the refinement of retrieved information, optimizing the extraction of key insights and minimizing the inclusion of non-essential elements, thereby enhancing the utilization of retrieved data.

CRAG is plug-and-play and experimentally implemented into RAG (Lewis et al., 2020) and Self-RAG (Asai et al., 2023) for demonstrating its adaptability to RAG-based approaches. Results on four datasets of PopQA (Mallen et al., 2023), Biography (Min et al., 2023), Pub Health (Zhang et al., 2023a), and Arc-Challenge (Bhakthavatsalam et al., 2021) show that CRAG can significantly improve the performance of standard RAG and state-of-the-art Self-RAG, demonstrating its generalizability across both short- and long-form generation tasks. To facilitate others to reproduce our results, we will publish all source code later.

In summary, our contributions in this paper are three-fold: 1) This paper studies the scenarios where the retriever returns inaccurate results and, to the best of our knowledge, makes the first attempt to design corrective strategies for RAG to improve its robustness. 2) A plug-and-play method named CRAG is proposed to improve the ability of automatic self-correction and efficient utilization of retrieved documents. 3) Experimental results extensively demonstrate CRAG’s adaptability to RAG-based approaches and its generalizability across short- and long-form generation tasks.

## Related Work

### Hallucinations of LLMs
Although LLMs have exhibited impressive abilities to understand instructions and generate fluent language texts (Bang et al., 2023; Qin et al., 2023; Zhong et al., 2023), one of the most severe issues that LLMs have still been struggling with is hallucinations. As many studies found (Zhang et al., 2023b; Shuster et al., 2021), either outdated information or incorrect knowledge that is activated would seriously result in hallucinations. Large-scale unregulated training data collection, low proportion of high-quality sampling data, imperfection of data allocation in the input space, and many other realistic factors could impact the LLMs and exacerbate the problems. Thus, it is obvious that the lack of accurate and specific knowledge can lead to misleading or even inaccurate generation, which will severely hurt the experience of users in most practical applications.

### Retrieval-Augmented Generation
RAG (Lewis et al., 2020; Guu et al., 2020) is regarded as a useful method to address the issues above, which enhances the input questions of generative LMs with retrieved documents. It usually provides an extra knowledge source from a specific corpus, i.e., Wikipedia, which greatly improves the performance of LMs in a variety of tasks, especially in the knowledge-intensive ones. The proposed methods generally leverage information retrieval to supply documents containing relevant knowledge for generative LLMs. Earlier studies adopt either
---
#
# Document

# Improving Robustness of Response Generation in Language Models

In recent years, there have been advancements in approaches to enhance response generation in language models. One key aspect that has been explored is the integration of retrievers at the front end of pre-trained language models to improve the relevance and accuracy of generated responses. However, there is a critical question that arises - what happens if the retrieval process fails? This issue can lead to inaccuracies in the generated responses, highlighting the importance of ensuring the effectiveness of the retrieval system.

# Advanced RAG

Several advanced approaches have been developed based on the original Retrieval-Augmented Generation (RAG) model. Some of these approaches focus on selectively retrieving knowledge based on the query, while others aim to improve the robustness of the retrieval process. For example:

- Self-RAG (Asai et al., 2023) selectively retrieves knowledge and incorporates a critic model to determine the relevance of the retrieved information.
- Yoran et al. (2023) developed an NLI model to identify irrelevant context and enhance robustness.
- SAIL (Luo et al., 2023) is tuned to insert retrieved documents into instructions.
- Toolformer (Schick et al., 2023) is pre-trained to access external APIs like Wikipedia.

In long-text generation tasks, the timing and relevance of retrieval become crucial. Jiang et al. (2023) have proposed methods to anticipate future content and optimize the retrieval process for long-form generation. Compared to recent studies, this paper focuses on addressing scenarios where the retriever returns inaccurate results and aims to design corrective strategies to enhance the robustness of response generation.

# Task Formulation

Building on previous work, the system is designed to generate an output Y based on input X and a corpus of knowledge documents C = {d1, ..., dN}. The framework consists of a retriever R and a generator G, where the retriever retrieves relevant documents D = {dr1, ..., drk} from the corpus C based on the input X. The generator then uses the input X and retrieved documents D to generate the output Y. The framework can be represented as:

P(Y|X) = P(D|X)P(Y, D|X) (Equation 1)

The seamless coupling of the retriever and generator highlights the importance of accurate retrieval in ensuring satisfactory responses from the system.

# CRAG

# Overview of Model Inference

CRAG introduces corrective strategies to enhance the robustness of response generation. At inference, a lightweight retrieval evaluator is used to assess the relevance of retrieved documents to the input query. The relevance score triggers actions such as refinement of retrieved documents, discarding incorrect retrievals, or resorting to web searches for corrections. The goal is to optimize the retrieval results before utilizing a generative model.

# Retrieval Evaluator

The accuracy of the retrieval evaluator is crucial in determining the quality of retrieved documents. By fine-tuning models like T5-large, the system aims to correct irrelevant retrievals and improve overall system performance.
---
# Figure 2: An overview of CRAG at inference.

A retrieval evaluator is constructed to evaluate the relevance of the retrieved documents to the input, and estimate a confidence degree based on which different knowledge retrieval actions of {Correct, Incorrect, Ambiguous} can be triggered.

Every question, there are generally 10 documents retrieved. The question is concatenated with each single document as the input, and the evaluator predicts the relevance score for each question-document pair individually. We also tried to prompt ChatGPT to identify the retrieval relevance for comparison, but it underperforms as elaborated in Section 5.5. Based on these calculated relevance scores, a final judgment is made as to whether the retrieval is correct or not associated with the action trigger. Compared with the critic model of Self-RAG (Asai et al., 2023) that instruction-tuned LLaMA-2 (7B), the evaluator designed in CRAG demonstrates the advantages of being quite lightweight (0.77B).

# 4.3 Action Trigger

To correct the irrelevant documents and refine the target documents as needed, actions should be executed discriminately. Based on the aforementioned confidence score for each retrieved document, three types of actions are designed and triggered accordingly where the upper and lower thresholds are set. If the confidence score is higher than the upper threshold, the retrieved document is identified as Correct, while identified as Incorrect if below the lower threshold. Otherwise, Ambiguous is executed. Each retrieved document is conducted individually and integrated eventually.

# Correct

Here, a retrieval is assumed Correct when the confidence score of at least one retrieved document is higher than the upper threshold. If so, it means that there are relevant documents in the retrieved results. Even if a relevant document can be found, there is inevitably some noisy knowledge strips in this document. To extract the most critical knowledge strips within this document, a knowledge refinement method is further designed which will be elaborated in Section 4.4.

# Incorrect

Besides, a retrieval is assumed Incorrect when the confidence scores of all retrieved documents are below the lower threshold. This indicates that all retrieved documents are considered irrelevant, which are unhelpful for generation. Therefore, we need to seek new sources of knowledge for correction. Here, web search is introduced to search from the Internet as
---
# Algorithm 1: CRAG Inference

Require: E (Retrieval Evaluator), W (Query Rewriter), G (Generator)

Input: x (Input question), D = {d1, d2, ..., dk} (Retrieved documents)

Output: y (Generated response)

1. scorei = E evaluates the relevance of each pair (x, di), di ∈ D
2. Confidence = Calculate and give a final judgment based on {score1, score2, ...scorek}

// Confidence has 3 optional values: [CORRECT], [INCORRECT] or [AMBIGUOUS]
3. if Confidence == [CORRECT] then
4. 1. Internal_Knowledge = Knowledge_Refine(x, D)
2. k = Internal_Knowledge

else if Confidence == [INCORRECT] then
5. 1. External_Knowledge = Web_Search(W Rewrites x for searching)
2. k = External_Knowledge

else if Confidence == [AMBIGUOUS] then
6. 1. Internal_Knowledge = Knowledge_Refine(x, D)
2. External_Knowledge = Web_Search(W Rewrites x for searching)
3. k = Internal_Knowledge + External_Knowledge

G predicts y given x and k

Elaborated in Section 4.5. This corrective action helps overcome the embarrassing challenge where no reliable knowledge can be referred to.

Ambiguous: Except for the above two situations, the remaining will be assigned to an intermediate action of Ambiguous. Since the retrieval evaluator is not confident in its judgment, both types of processed knowledge in Correct and Incorrect are combined to complement each other. Implementing such a moderating and soft strategy can significantly contribute to strengthening the robustness and resilience of the system, fostering a more adaptable framework for optimal performance.

# 4.4 Knowledge Refinement

Given a retrieved relevant document, a decompose-then-recompose knowledge refinement method is designed to further extract the most critical knowledge strips in it. First, each retrieved document is segmented into fine-grained knowledge strips through heuristic rules, more details are available in Appendix B.2. Then, the retrieval evaluator fine-tuned in Section 4.2 is employed to calculate the relevance score of each knowledge strip. Based on these scores, irrelevant knowledge strips are filtered out, while relevant ones are recomposed via concatenation in order, namely internal knowledge.

# 4.5 Web Search

It is extremely important to seek complementary external knowledge if the retrieved results are all assumed irrelevant. Since retrieval from static and limited corpora can only return sub-optimal documents in terms of scope and diversity, large-scale web searches (Piktus et al., 2021; Komeili et al., 2022) are integrated as a strategic extension of RAG. Specifically, the inputs are rewritten into queries composed of keywords by ChatGPT to mimic the daily usage of search engine. The prompt for rewriting is shown in Appendix A. In CRAG, a public and accessible commercial web search API is adopted to generate a series of URL links for every query. Moreover, we utilize the URL links to navigate web pages, transcribe their content, and employ the same knowledge refinement method as Section 4.4 to derive the relevant web knowledge, namely external knowledge.

# 5 Experiments

We conducted experiments to extensively demonstrate CRAG’s adaptability to RAG-based approaches and its generalizability across both short- and long-form generation tasks.

# 5.1 Tasks, Datasets and Metrics

CRAG was evaluated on four datasets, including PopQA (Mallen et al., 2023) (short-form generation), Biography (Min et al., 2023) (long-form generation), PubHealth (Zhang et al., 2023a) (true-or-false question), and Arc-Challenge (Bhakthavatsalam et al., 2021) (multiple-choice question).

2In this study, Google Search API is utilized for searching.
---
|Method|PopQA (Accuracy)|Bio (FactScore)|Pub (Accuracy)|ARC (Accuracy)|
|---|---|---|---|---|
|LLaMA2-c13B|20.0|55.9|49.4|38.4|
|Ret-LLaMA2-c13B|51.8|79.9|52.1|37.9|
|ChatGPT|29.3|71.8|70.1|75.3|
|Ret-ChatGPT|50.8|-|54.7|75.3|
|Perplexity.ai|-|71.2|-|-|
|LLaMA27B|14.7|44.5|34.2|21.8|
|Alpaca7B|23.6|45.8|49.8|45.0|
|LLaMA213B|14.7|53.4|29.4|29.4|
|Alpaca13B|24.4|50.2|55.5|54.9|
|CoVE65B|-|71.2|-|-|
|LLaMA27B|38.2|78.0|30.0|48.0|
|Alpaca7B|46.7|76.6|40.2|48.0|
|SAIL|-|-|69.2|48.4|
|LLaMA213B|45.7|77.5|30.2|26.0|
|Alpaca13B|46.1|77.7|51.1|57.6|
|LLaMA2-hf-7b|RAG|37.7|44.9|9.1|23.8|
|CRAG|39.8|47.7|9.1|25.8|
|Self-RAG*|29.0|32.2|0.7|23.9|
|Self-CRAG|49.0|69.1|0.6|27.9|
|SelfRAG-LLaMA2-7b|RAG|40.3|59.2|39.0|46.7|
|CRAG|59.3|74.1|75.6|54.8|
|Self-RAG|54.9|81.2|72.4|67.3|
|Self-CRAG|61.8|86.2|74.8|67.2|

Table 1: Overall evaluation results on the test sets of four datasets. Results are separated based on the generation LLMs. Bold numbers indicate the best performance among all methods and LLMs. Gray-colored bold scores indicate the best performance using a specific LLM. * indicates the results reproduced by us, otherwise results except ours are cited from their original papers.

Following previous work, accuracy was adopted as the evaluation metric for PopQA, PubHealth, and Arc-Challenge. FactScore (Min et al., 2023) was adopted as the evaluation metric for Biography. Readers can refer to Appendix B.1 for more details.

# Baselines

We primarily compared CRAG with both approaches without and with retrieval, where the latter consists of standard RAG and advanced RAG.

# Baselines without retrieval

We evaluated some public LLMs, LLaMA2-7B,13B (Touvron et al., 2023b), instruction-tuned models, Alpaca-7B,13B (Dubois et al., 2023), and CoVE65B (Dhuliawala et al., 2023) which introduces iterative engineering to improve the factuality of LLM generations. Propriety LLMs such as LLaMA2-chat13B and ChatGPT are also included.

# Advanced RAG

1. SAIL (Luo et al., 2023) that instruction-tuned an LM on the Alpaca instruction-tuning data with top retrieved documents inserted before instructions.

2. Self-RAG (Asai et al., 2023) that tuned the LLaMA2 on the instruction-tuning data containing several sets of reflection tokens which were labeled by GPT-4 (OpenAI, 2023).

3. Following Asai et al. (2023), we also
---
cited the results of retrieval-augmented baselines
trained with private data: Ret-ChatGPT and Ret-
LLaMA-chat, which deploy the same augmenta-
tion technique above, as well as perplexity.ai, an
InstructGPT-based production search system.

# Results

Table 1 presents the results on four datasets. The
model coupling the proposed method with standard
RAG is named CRAG and that coupling with Self-
RAG is named Self-CRAG. Readers can refer to
Appendix B.2 for more implementation details of
our proposed methods. From these results, we can
conclude the following findings:

- First, the proposed method can significantly
improve the performance of RAG and Self-RAG.
Specifically, CRAG outperformed RAG by
margins of 19.0% accuracy on PopQA, 14.9%
FactScore on Biography, 36.6% accuracy on
PubHealth, and 8.1% accuracy on Arc-Challenge
when based on SelfRAG-LLaMA2-7b, as well
as by margins of 2.1% accuracy on PopQA,
2.8% FactScore on Biography, and 2.0% on
Arc-Challenge when based on LLaMA2-hf-7b.
Compared with the current state-of-the-art Self-
RAG, Self-CRAG outperformed it by margins of
20.0% accuracy on PopQA, 36.9% FactScore on
Biography, and 4.0% accuracy on Arc-Challenge
when based on LLaMA2-hf-7b, as well as by
margins of 6.9% accuracy on PopQA, 5.0%
FactScore on Biography, and 2.4% accuracy on
PubHealth, when based on SelfRAG-LLaMA2-7b.
These results demonstrated the adaptability
of CRAG which is plug-and-play and can be
implemented into RAG-based approaches.
- Second, the proposed method demonstrated
great generalizability across a variety of gen-
eration tasks. In particular, these benchmarks
reported in Table 1 respectively represent different
practical scenarios including short-form entity
generation (PopQA), long-form generation (Bi-
ography), and closed-set tasks (PubHealth, Arc-
Challenge). These results verified the consistent
effectiveness of CRAG. Its versatility across a spec-
trum of tasks underscores its robust capabilities and
generalizability across diverse scenarios.
- Third, the proposed method exhibited greater
flexibility in replacing the underlying LLM gen-
erator. It can be seen that CRAG still showed
competitive performance when the underlying
LLMs was changed from SelfRAG-LLaMA2-7b
to LLaMA2-hf-7b, while the performance of Self-RAG dropped significantly, even underperforming
the standard RAG on several benchmarks. The
reason for these results is that Self-RAG needs to be
instruction-tuned using human or LLM annotated
data to learn to output special critic tokens as
needed, while this ability is not learned in common
LLMs. CRAG does not have any requirements
for this ability. As you can imagine, when more
advanced LLMs are available in the future, they
can be coupled with CRAG easily, while additional
instruction tuning is still necessary for Self-RAG.

# Ablation Study

The impact of each triggered action. To fur-
ther verify the effectiveness of triggered actions
designed in the retrieval evaluator, ablation tests
for removing each single action in the proposed
method were conducted as shown in Table 2.
Evaluations on the PopQA dataset were conducted
to demonstrate the performance change in terms of
accuracy. Specifically, when the action Correct
or Incorrect was removed, it was merged with
Ambiguous so that the proportion that originally
triggered Correct or Incorrect would trigger
Ambiguous. On the other hand, when the action
Ambiguous was removed, there was only one
threshold against which all input queries clearly
triggered Correct or Incorrect. From these
results, it can be seen that there was a performance
drop no matter which action was removed, illustrat-
ing that each action contributed to improving the
robustness of generation.

The impact of each knowledge utilization oper-
ation. Table 3 illustrated how the performance
changed if a key knowledge utilization operation
was ablated. Evaluations on the PopQA dataset in
terms of accuracy were conducted by individually
removing the knowledge utilization operations of
document refinement, search query rewriting, and
---
# LLaMA2-hf-7b SelfRAG-LLaMA2-7b

| |CRAG| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| |47.3| |59.3|60| | | | |
|w/o. refinement|38.9| |47.0|50| | | | |
|w/o. rewriting|44.8| |56.6| | | | | |
|w/o. selection|44.0| |53.8|40| | | | |

|Self-CRAG| | | | | | | |
|---|---|---|---|---|---|---|---|
| |49.0| |61.8| | | | |
|w/o. refinement|35.9| |52.2| | | | |
|w/o. rewriting|37.2| |58.4| | | | |
|w/o. selection|24.9| |57.9| | | | |

Table 3: Ablation study for removing each knowledge utilization operation on the PopQA in terms of accuracy.

Accuracy

|Our Retrieval Evaluator (T5-based)|84.3|
|---|---|
|ChatGPT|58.0|
|ChatGPT-CoT|62.4|
|ChatGPT-few-shot|64.7|

Table 4: Evaluation of our retrieval evaluator and ChatGPT for the retrieval results on the PopQA dataset.

external knowledge selection. Removing document refinement denoted that the original retrieved documents were directly fed to the following generator, as in most existing works. Additionally, removing search query rewriting denoted that questions were not rewritten into queries consisting of keywords during knowledge searching. Eventually, removing knowledge selection denoted that all searched content of web pages was all regarded as the external knowledge without selection. These results help derive the findings that the performance of the final system degraded no matter which knowledge utilization operation was removed, revealing that each knowledge utilization operation contributed to improving the utilization of knowledge.

# 5.5 Accuracy of the Retrieval Evaluator

The quality of the retrieval evaluator significantly determined the performance of the entire system. Given the document retrieval results, we assessed whether the retrieval evaluator can accurately determine the overall quality of these results. The assessment accuracy on the PopQA dataset of our retrieval evaluator and the commercial LLM ChatGPT on the document retrieval results was shown in Table 4. The prompts of ChatGPT, ChatGPT-CoT, and ChatGPT-few-shot used in our experiments can be referred to in Appendix A. Results reveal that the lightweight T5-based retrieval evaluator significantly outperformed the competitive ChatGPT in all settings.

Figure 3: The generation performance of Self-RAG, Self-CRAG, RAG, and CRAG given different retrieval performance on the PopQA dataset with SelfRAG-LLaMA-7b. The lower horizontal line demonstrates the performance of the generator without retrieval.

# 5.6 Robustness to Retrieval Performance

To further verify the robustness of the proposed method to retrieval performance, we studied how the generation performance changed given different retrieval performance. A part of accurate retrieval results were deliberately removed at random to imitate a low-quality retriever and evaluate how the performance changed. Figure 3 demonstrated the performance change of Self-RAG and Self-CRAG on the PopQA dataset. It can be seen that the generation performance of Self-RAG and Self-CRAG dropped as the retrieval performance dropped, indicating that the generator relied heavily on the quality of the retriever. Furthermore, as the retrieval performance dropped, the generation performance of Self-CRAG dropped more slightly than that of Self-RAG. These results imply the superiority of Self-CRAG over Self-RAG on enhancing the robustness to retrieval performance.

# 6 Conclusion

This paper studies the problem where RAG-based approaches are challenged if retrieval goes wrong, thereby exposing inaccurate and misleading knowledge to generative LMs. Corrective Retrieval Augmented Generation is proposed to improve the robustness of generation. Essentially, a lightweight retrieval evaluator is to estimate and trigger three knowledge retrieval actions discriminately. With the further leverage of web search and optimized knowledge utilization, CRAG has significantly improved the ability of automatic self-correction and efficient utilization of retrieved documents. Experiments extensively demonstrate its adaptability to RAG-based approaches as well as generalizability across short- and long-form generation tasks.
---
#
# References

# Limitations

Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2023. Palm: Scaling language modeling with pathways. J. Mach. Learn. Res., 24:240:1–240:113.

While we primarily proposed to improve the RAG framework from a corrective perspective, how to detect and correct the wrong knowledge more accurately and effectively still requires further study. Although CRAG can be seamlessly coupled with various RAG-based approaches, fine-tuning a retrieval evaluator is inevitable. In addition, potential bias introduced by web searches is also worth concern. The quality of internet sources can vary significantly, and incorporating such data without enough consideration may introduce noise or misleading information to the generated outputs.

Future work will further explore a more stable and reliable method of retrieval augmentation.

# References

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, et al. 2023. PaLM 2 technical report. CoRR, abs/2305.10403.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023. Self-rag: Learning to retrieve, generate, and critique through self-reflection. CoRR, abs/2310.11511.

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan Xu, and Pascale Fung. 2023. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. CoRR, abs/2302.04023.

Sumithra Bhakthavatsalam, Daniel Khashabi, Tushar Khot, Bhavana Dalvi Mishra, Kyle Richardson, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord, and Peter Clark. 2021. Think you have solved direct-answer question answering? try arc-da, the direct-answer AI2 reasoning challenge. CoRR, abs/2102.03315.

Tom B Brown, Benjamin Mann, Nick Ryder, et al. 2020. Language models are few-shot learners. In Advances in neural information processing systems, pages 1877–1901.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben. 2023. Retrieval augmented language model pre-training. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 3929–3938. PMLR.

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised dense information retrieval with contrastive learning. Trans. Mach. Learn. Res., 2022.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Comput. Surv., 55(12):248:1–248:38.

Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 7969–7992. Association for Computational Linguistics.

Mojtaba Komeili, Kurt Shuster, and Jason Weston. 2022. Internet-augmented dialogue generation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 8460–8478. Association for Computational Linguistics.
---
#
# References

# References

Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural
Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Huayang Li, Yixuan Su, Deng Cai, Yan Wang, and Lemao Liu. 2022. A survey on retrieval-augmented text generation. CoRR,
abs/2202.01110.

Hongyin Luo, Tianhua Zhang, Yung-Sung Chuang, Yuan Gong, Yoon Kim, Xixin Wu, Helen Meng, and James R. Glass. 2023. Search
augmented instruction learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore,
December 6-10, 2023, pages 3717–3729. Association for Computational Linguistics.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When not to trust
language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings of the 61st
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada,
July 9-14, 2023, pages 9802–9822. Association for Computational Linguistics.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh
Hajishirzi. 2023. Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. Proceedings
of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023,
pages 12076–12100. Association for Computational Linguistics.

Dor Muhlgay, Ori Ram, Inbal Magar, Yoav Levine, Nir Ratner, Yonatan Belinkov, Omri Abend, Kevin Leyton-Brown, Amnon
Shashua, and Yoav Shoham. 2023. Generating benchmarks for factuality evaluation of language models. CoRR, abs/2307.06908.

Chengwei Qin, Aston Zhang, Zhuosheng Zhang, Jiaao Chen, Michihiro Yasunaga, and Diyi Yang. 2023. Is chatgpt a general-purpose
natural language processing task solver? In Proceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 1339–1384. Association for Computational Linguistics.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J.
Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res.,
21:140:1–140:67.

Md. Rashad Al Hasan Rony, Ricardo Usbeck, and Jens Lehmann. 2022. Dialokg: Knowledge-structure aware task-oriented dialogue
generation. In Findings of the Association for Computational Linguistics: NAACL 2022, Seattle, WA, United States, July
10-15, 2022, pages 2557–2571. Association for Computational Linguistics.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas
Scialom. 2023. Toolformer: Language models can teach themselves to use tools. CoRR, abs/2302.04761.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Schärli, and Denny Zhou. 2023.
Large language models can be easily distracted by irrelevant context. In Proceedings of the 40th International Conference
on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 31210–31227. PMLR.

Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021. Retrieval augmentation reduces hallucination
in conversation. In Findings of the Association for Computational Linguistics: EMNLP 2021, Virtual Event / Punta Cana,
Dominican Republic, 16-20 November, 2021, pages 3784–3803. Association for Computational Linguistics.

OpenAI. 2023. GPT-4 technical report. CoRR,

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal,
Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter
Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human
feedback. In NeurIPS.

Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Dmytro Okhonko, Samuel Broscheit, Gautier Izacard, Patrick S. H.
Lewis, Barlas Oguz, Edouard Grave, Wen-tau Yih, and Sebastian Riedel. 2021. The web

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière,
Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a.
Llama: Open, and efficient foundation language models. CoRR abs/2302.13971.
---
- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. *CoRR*, abs/2307.09288.
- Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. 2023. Making retrieval-augmented language models robust to irrelevant context. *CoRR*, abs/2310.01558.
- Tianhua Zhang, Hongyin Luo, Yung-Sung Chuang, Wei Fang, Luc Gaitskell, Thomas Hartvigsen, Xixin Wu, Danny Fox, Helen Meng, and James R. Glass. 2023a. Interpretable unified language checking. *CoRR*, abs/2304.03728.
- Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. 2023b. Siren’s song in the AI ocean: A survey on hallucination in large language models. *CoRR*, abs/2309.01219.
- Qihuang Zhong, Liang Ding, Juhua Liu, Bo Du, and Dacheng Tao. 2023. Can chatgpt understand too? A comparative study on chatgpt and fine-tuned BERT. *CoRR*, abs/2302.10198.
---
#
# Task Prompts

# Task Prompts

Given a question, does the following document have exact information to answer the question? Answer yes or no

The prompts for generating knowledge keywords as web search queries were illustrated in Table 5.

|Question|Document|Answer|
|---|---|---|
|In what city was Abraham Raimbach born?|Bancroft was born on November 25, 1839 in New Ipswich, New Hampshire to James Bancroft and Sarah Kimball. At an early age he was cared for by Mr. and Mrs. Patch of Ashby, Massachusetts, the neighboring town. While not legally adopted, they named him Cecil Franklin Patch Bancroft, adding Franklin Patch after the son Mr. and Mrs. Patch had who recently died. He attended public schools in Ashby as well as the Appleton Academy in New Ipswich. He entered Dartmouth College in 1856 at the age of sixteen and graduated in 1860 near the top of his class. Bancroft continued his education as he began his career in teaching. He took classes at the Union Theological Seminary in New York City during the 1864-65 academic year. While there he was a member of the United States Christian Commission, traveling to support soldiers during the Civil War. He then transferred to the Andover Theological Seminary where he would graduate in 1867.|No|
|In what country is Wilcza Jama, Sokółka County?|Wilcza Jama is a village in the administrative district of Gmina Sokółka, within Sokółka County, Podlaskie Voivodeship, in north-eastern Poland, close to the border with Belarus.|Yes|

# The prompts to instruct ChatGPT as the evaluator were illustrated in Table 6, Table 7, and Table 8 respectively.

|Question|Document|Answer|
|---|---|---|
|[question]|[document]| |

# Table 6: The direct prompt to GPT-3.5 Turbo as the evaluator.

Given a question, does the following document have exact information to answer the question? Answer yes or no only.

Question: [question]

Document: [document]

Think Step by step, and answer with yes or no only.

Answer:

# Table 7: The prompt to GPT-3.5 Turbo with Chain-of-Thought as the evaluator.

Question: [question]

Document: [document]

Answer:

# Table 8: The few-shot prompt to GPT-3.5 Turbo as the evaluator.

Question: [question]

Document: [document]

Answer:
---
## Experiments

### Tasks, Datasets and Metrics

CRAG was evaluated on four datasets, which are in public domain and licensed for research purposes, including:

- **PopQA** (Mallen et al., 2023): A short-form generation task where only one entity of factual knowledge is expected to be answered for each single question. In our experiments, we followed the setting in Self-RAG (Asai et al., 2023) which evaluated methods on a long-tail subset consisting of 1,399 rare entity queries whose monthly Wikipedia page views are less than 100. Accuracy was adopted as the evaluation metric.

- **Biography** (Min et al., 2023): A long-form generation task tasked with generating a detailed biography about a certain entity. FactScore (Min et al., 2023) was adopted to evaluate the generated biographies.

- **PubHealth** (Zhang et al., 2023a): A task in the health care domain consisting of true-or-false questions. The model is tasked to verify the authenticity of claims about health with factual information and give a judgment. Accuracy was adopted as the evaluation metric.

- **Arc-Challenge** (Bhakthavatsalam et al., 2021): A multiple-choice question task about daily commonsense science phenomena. The model is required to select the correct description among 3 or 4 optional choices for a given scientific event. Accuracy was adopted as the evaluation metric.

### Implementation Details

**Retrieval Evaluator:** We fine-tuned the retrieval evaluator based on the lightweight T5-large (Raffel et al., 2020) pre-trained model. The same retrieval results through Contriever (Izacard et al., 2022) were provided by Self-RAG and adopted in our experiments. Relevance signals for fine-tuning the evaluator were collected from existing datasets. Negative samples were randomly sampled. The fine-tuned evaluator was transferred and utilized on the Bio, Pub, and ARC datasets during inference.

**Internal Knowledge:** Retrieved results were segmented into internal strips for fine-grained retrieval. The top-k was set to 5, and the filter threshold was set at -0.5.

**External Knowledge:** Google Search API was used to search for relevant URLs, with top-k set to 5. Pages from Wikipedia were added preferentially.

**Generator:** CRAG is a plug-and-play method, compatible with all generation models that can be used in RAG. LLaMA2 (Touvron et al., 2023b) was adopted for generation. LLaMA2-hf-7b from huggingface was used to generate responses.

For more details, refer to [PopQA dataset](https://huggingface.co/datasets/akariasai/PopQA).
---
On several tasks, we further utilized the launched model, SelfRAG-LLaMA2-7b, as a new generator to be consistent with their work and study the specific improvement of our method.

Self-CRAG: To demonstrate that our plug-and-play approach can be utilized in other concurrent studies, we specifically designed to insert our CRAG into the Self-RAG (Asai et al., 2023) framework and named it Self-CRAG. Self-RAG is an advanced RAG approach that introduces a critic model to decide whether to retrieve and which retrieved document to be referred for generation. It meets our demand for deciding which action to be triggered, thus we replaced the retrieved items in Self-RAG with our processed internal knowledge for Correct, external knowledge for Incorrect, and combined knowledge for Ambiguous.

# Results on PubHealth and Arc-Challenge

It is worth mentioning that the performance on PubHealth based on LLaMA2-hf-7b was much worse than others. We studied these cases and found that LLaMA2-hf-7b is relatively weak in instruction comprehension. Most of the cases fail to generate True or False in such a binary-question task, resulting in a low accuracy during the evaluation. This situation somewhat happens in Arc-Challenge as well, when the model is tasked to generate the index of a candidate.