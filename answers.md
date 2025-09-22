# Part C: Short Answer Questions

## Question 1: Limited Data Scenario
**If you only had 200 labeled replies, how would you improve the model without collecting thousands more?**

With only 200 labeled samples, I would focus on transfer learning and data augmentation strategies. First, I'd leverage pre-trained transformer models like DistilBERT which already understand language patterns from massive datasets, requiring minimal fine-tuning on our small dataset. For data augmentation, I'd use techniques like back-translation (translate to another language and back), paraphrasing with language models, and synonym replacement to create label-preserving variations. Additionally, I'd implement few-shot learning approaches using large language models to generate synthetic examples for underrepresented classes, apply strong regularization techniques like dropout and early stopping, and use cross-validation to maximize the use of available data.

## Question 2: Bias and Safety in Production
**How would you ensure your reply classifier doesn't produce biased or unsafe outputs in production?**

To prevent biased or unsafe outputs, I would implement a multi-layered approach starting with comprehensive bias testing across different demographic groups, industries, and communication styles during development. In production, I'd establish continuous monitoring of model predictions and class distributions, implement confidence thresholds where low-confidence predictions are flagged for human review, and maintain detailed logging for auditability. I'd also create feedback loops where human reviewers can correct predictions, regularly retrain the model with corrected data, and establish clear escalation procedures for handling problematic outputs. Additionally, I'd conduct regular bias audits and maintain diverse training data to ensure fair representation across different groups.

## Question 3: LLM Prompt Design for Personalized Cold Emails
**What prompt design strategies would you use for generating personalized cold email openers with an LLM?**

For effective personalized cold email generation, I would design prompts with explicit constraints and structured inputs. The prompt would include clear instructions about tone (professional, friendly), length limits (1-2 sentences), and required personalization elements like company name, recipient role, and recent company news or achievements. I'd use few-shot learning by providing 3-5 high-quality examples of personalized openers with different scenarios, include negative examples showing what to avoid (generic phrases, overly salesy language), and structure the prompt to require specific personalization tokens that must be filled. The prompt would also include guidelines for maintaining authenticity, avoiding spam-like language, and ensuring the opener relates directly to the recipient's potential pain points or interests based on the provided context.

---

## Additional Technical Considerations

### Model Selection Criteria
When choosing between the baseline and transformer models for production, consider:

- **Latency Requirements**: Baseline model (~1-2ms) vs Transformer (~20-50ms)
- **Accuracy Needs**: Baseline (~85-90%) vs Transformer (~90-95%)
- **Resource Constraints**: Baseline (10MB, CPU-friendly) vs Transformer (250MB, GPU-preferred)
- **Interpretability**: Baseline offers feature importance, Transformer is more black-box

### Deployment Recommendations

1. **High-Volume, Low-Latency**: Use baseline model with caching
2. **High-Accuracy Requirements**: Use transformer with GPU acceleration
3. **Hybrid Approach**: Use baseline for initial filtering, transformer for uncertain cases
4. **A/B Testing**: Deploy both models and compare real-world performance

### Monitoring and Maintenance

- Track prediction confidence distributions over time
- Monitor for data drift in incoming email patterns
- Implement automated retraining pipelines
- Maintain human-in-the-loop validation for edge cases
- Regular model performance audits and bias assessments

### Scalability Considerations

- Implement model serving with load balancing
- Use batch processing for non-real-time predictions
- Consider model quantization for reduced memory usage
- Implement caching for repeated similar queries
- Plan for horizontal scaling with containerization