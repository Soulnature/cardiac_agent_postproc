# Method Section — MICCAI Paper Draft

## Multi-Agent LLM Framework for Anatomy-Aware Post-Processing of Cardiac MRI Segmentation

---

## 2. Method

# Method Section — MICCAI Paper Draft

## Multi-Agent LLM Framework for Anatomy-Aware Post-Processing of Cardiac MRI Segmentation

---

## 2. Method

### 2.1 Problem Formulation

We formulate the cardiac segmentation refinement task as a **Sequential Decision Process** without ground truth, where a team of specialized agents interacts with an environment $\mathcal{E}$ consisting of the raw MR image $I$ and a candidate segmentation mask $\mathbf{M}$.

Let $S_t = (\mathbf{M}_t, H_t)$ denote the state at step $t$, comprising the current mask and the history of diagnostic findings $H_t$. The action space $\mathcal{A}$ encompasses a discrete set of topological and morphological repair operations (e.g., `close_holes`, `atlas_transfer`). The transition function $\mathcal{T}: S_t \times a_t \rightarrow S_{t+1}$ is deterministic, executing the chosen repair. The objective is to maximize a reference-free Reward Quality Score, $R(\mathbf{M})$, which proxies anatomical correctness:

$$ \mathbf{M}^* = \arg\max_{\mathbf{M}} R(\mathbf{M} \mid \mathcal{K}), $$

where $\mathcal{K}$ represents external knowledge (Atlases, Visual Knowledge Base). Unlike single-agent RL, we decompose the policy $\pi$ into a **hierarchical multi-agent system** to handle the complexity of identifying valid repairs in a high-dimensional space.

### 2.2 Multi-Agent System Framework

Our framework (Fig. 1) orchestrates six specialized agents using Large Language Models (LLMs) for reasoning and Vision-Language Models (VLMs) for visual grounding. The system operates in a dynamic feedback loop managed by a central Coordinator.

#### 2.2.1 Coordinator Agent: Orchestration & Policy Control
The **Coordinator Agent** functions as the meta-controller, managing the global state $S_t$ through a **finite state machine**. It implements a **dynamic retry policy** that adapts based on feedback:
*   **Dispatch**: Routes cases to Triage and, if flagged, initializes the Repair Sub-team.
*   **Feedback-Driven Loop**: Upon verification failure, the Coordinator captures the rejection reason $r_{fail}$ (e.g., "new artifacts introduced") and appends it to the context history $H_{t+1}$. This feedback is explicitly injected into the next Diagnosis prompt, enabling the system to "learn from mistakes" within the episode.
*   **Termination**: The loop terminates upon success or when a maximum of $N_{max}=3$ adaptive retries is reached. To ensure safety, the system reverts to the original prediction $\mathbf{M}_0$ if the final RQS does not improve by a margin $\delta > 0$.

#### 2.2.2 Triage Agent: Efficient Gating
To optimize computational resources, the **Triage Agent** acts as a lightweight gatekeeper, filtering out clinically acceptable masks.
*   **Feature Extraction**: It computes a vector $\mathbf{f} \in \mathbb{R}^{30}$ of handcrafted geometric descriptors, including topological Euler numbers, connected component counts per class, normalized area fractions, and intensity-based edge consistency metrics.
*   **Hybrid Classification**: A Random Forest classifier first predicts a defect probability $p_{defect}$. For ambiguous cases ($0.4 < p_{defect} < 0.6$), the agent escalates to a VLM for visual logical confirmation.
*   **Outcome**: Only cases with confirmed high defect probability enter the computationally intensive repair loop, effectively reducing the LLM inference cost by $\sim 60\%$ on standard datasets.

#### 2.2.3 Diagnosis Agent: Spatial & Anatomical Reasoning
The **Diagnosis Agent** constructs the "Observation" $O_t$ for the planner decision. It employs a **multimodal synthesis** strategy:
*   **Rule-Based Reasoning**: A deterministic module scans for 15+ axiomatic violations, mapping them to anatomical quadrants relative to the LV centroid. Key checks include:
    *   *Topology*: RV-LV connectivity (medically impossible), fragmented myocardial rings.
    *   *Inter-Slice Consistency*: Abrupt centroid shifts ($>3\sigma$) or area changes ($>25\%$) along the z-axis.
*   **Visual Reasoning**: A VLM (e.g., GPT-4o) inspects color-coded mask overlays to detect subtle shape anomalies like "basal leakage" or "apical thinning" that elude simple heuristic rules.
**Output:** A structured `SpatialDiagnosis` list (e.g., *[{Type: Gap, Location: Basal-Septal, Severity: High}, {Type: Island, Location: RV-Apical, Severity: Low}]*), grounding the problem in precise anatomical semantics.

#### 2.2.4 Planner Agent: Strategic Sequencing
The **Planner Agent** generates the "Policy" $\pi(a_t | O_t)$. It maps the diagnosis list to an executable plan $\mathbf{P} = [a_1, ..., a_n]$.
*   **Dependency Resolution**: Enforces logical ordering constraints (e.g., *topological repairs* $\prec$ *atlas transfer* $\prec$ *morphological smoothing*) to prevent error propagation.
*   **Conflict Resolution**: Uses LLM reasoning to arbitrate between mutually exclusive repairs. For instance, if *RV-Dilate* and *LV-Dilate* are both suggested but would cause a collision, the Planner prioritizes the action corresponding to the higher-severity diagnosis.

#### 2.2.5 Executor Agent: Deterministic Action Space
The **Executor Agent** implements the transition function $\mathcal{T}$. It wraps a library of $>20$ atomic operations into a tailored action space:
*   **Context-Aware Repair**: Beyond standard morphology (erosion/dilation), the agent utilizes **Patient-Specific Atlas Transfer**. This mechanism registers a high-quality mask from a neighboring slice $z \pm 1$ using phase correlation and warps it to the current slice, effectively transferring valid anatomical shape priors.
*   **Probabilistic Atlas Repair**: For slices lacking valid neighbors, it queries the Population Probabilistic Atlas (Sec. 2.3) based on view and slice position to guide reconstruction.
*   **Atomic Verification**: Executes a "survival check" after each action $a_i$ to ensure the new state $\mathbf{M}_{t+1}$ satisfies basic validity constraints (e.g., area $\in [0.5A_{prev}, 2.0A_{prev}]$), safeguarding against catastrophic model hallucinations.

#### 2.2.6 Verifier Agent: Visual Turing Test
The **Verifier Agent** estimates the value function $V(S_t)$ via a "Visual Turing Test".
*   **Visual-Semantic Evaluation**: Compares state $S_t$ (before) and $S_{t+1}$ (after) using side-by-side overlays and a difference heatmap.
*   **Knowledge-Enhanced Reasoning**: Queries a **Visual Knowledge Base (VKB)** containing few-shot examples of successful vs. failed repairs. The VLM evaluates whether the change represents a genuine anatomical improvement (e.g., "restored myocardial continuity") versus a numerical hack (e.g., "smoothed noise without structural fix").
*   **Decision**: Outputs a binary decision $d \in \{Accept, Reject\}$ along with a rationale, which feeds back into the Coordinator's loop.

### 2.3 Knowledge Augmentation

The agents are augmented with static knowledge sources $\mathcal{K}$:
1.  **Probabilistic Shape Atlas**: A population-level shape prior constructed via K-Means clustering of affine-aligned training masks, enabling "Atlas Transfer" actions for gross defects.
2.  **Reward Quality Score (RQS)**: A differentiable-free objective function acting as the environment's intrinsic reward signal, aggregating topological penalties, data fidelity terms (edge alignment), and shape priors.
