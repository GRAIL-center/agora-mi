# AutoInterp Feature Gallery

## Layer 20

### Protected Groups

Faithfulness score: 1.000
Ranking family: layer_unique
Best proxy: interpretability
Rationale: The feature activates when the text mentions groups protected from discrimination based on sex, gender identity, or sexual orientation.
Boundary: The feature should stay low when the text discusses general demographics or does not mention specific protected groups.

### Even when

Faithfulness score: 1.000
Ranking family: policy_specific
Best proxy: discrimination
Rationale: The feature activates when a text expresses a contrast or exception using the phrase "even when".
Boundary: The feature should stay low when the text does not contain the phrase "even when" or when it is used in a non-contrastive context.

### Training, Test, and Verification Data

Faithfulness score: 1.000
Ranking family: layer_unique
Best proxy: bias
Rationale: The feature activates when the text mentions the distinct datasets used in machine learning: training, test, and verification data.
Boundary: The feature should stay low when the text discusses general data concepts or does not specifically refer to these three types of datasets.

### efficiency

Faithfulness score: 1.000
Ranking family: layer_unique
Best proxy: discrimination
Rationale: The feature activates when the text mentions improving or measuring efficiency.
Boundary: The feature should stay low when the text discusses fairness, bias, or other ethical considerations related to AI.

### Singapore's AI Strategy

Faithfulness score: 0.833
Ranking family: policy_specific
Best proxy: bias
Rationale: The feature activates when the text discusses Singapore's policies and initiatives related to artificial intelligence.
Boundary: The feature should stay low when the text focuses on general AI concepts or discusses AI in other countries.

## Layer 24

### Regularly evaluating claims

Faithfulness score: 0.667
Ranking family: policy_specific
Best proxy: transparency
Rationale: The feature activates when the text mentions the regular evaluation of claims made by vendors or entities regarding their AI offerings.
Boundary: The feature should stay low when the text discusses general principles of AI development or use without focusing on the evaluation of specific claims.

### Bring value

Faithfulness score: 0.667
Ranking family: policy_specific
Best proxy: interpretability
Rationale: The feature activates when the text mentions the concept of bringing value to individuals, communities, or society.
Boundary: The feature should stay low when the text discusses technical aspects of AI development or focuses on risks and limitations without mentioning value.

### California Medical Privacy Laws

Faithfulness score: 0.667
Ranking family: policy_specific
Best proxy: bias
Rationale: The feature activates when the text discusses California's laws regarding patient privacy and medical information.
Boundary: The feature should stay low when the text discusses general AI safety or national security applications of AI.

### Strategic Priority

Faithfulness score: 0.667
Ranking family: policy_specific
Best proxy: interpretability
Rationale: The feature activates when the text mentions a strategic priority within a policy document.
Boundary: The feature should stay low when the text does not discuss policy priorities or focuses on general topics unrelated to policy.

### 2 to 4 words

Faithfulness score: 0.500
Ranking family: layer_unique
Best proxy: transparency
Rationale: The feature activates when the text mentions regulations or requirements for hospitals or healthcare facilities.
Boundary: The feature should be low when the text discusses general concepts of AI or technology without a healthcare context.
