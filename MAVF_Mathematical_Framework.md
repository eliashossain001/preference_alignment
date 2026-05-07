# Multi-Agent Verification Framework for RLHF Preference Data
### A Rigorous Mathematical Treatment

---

## 1. Problem Setting

Let $\mathcal{D} = \{x^{(m)}\}_{m=1}^{M}$ be a dataset of human preference pairs, where each element is a triple $x = (q, r_+, r_-)$ consisting of a user prompt $q$, a human-preferred response $r_+$, and a rejected response $r_-$.

**Definition 1 (Latent Validity Variable).** Each preference pair $x$ is associated with a latent random variable

$$Z_x \in \{0, 1\},$$

where $Z_x = 1$ denotes that the pair represents valid supervision (the annotator's preference is consistent, grounded, and unbiased) and $Z_x = 0$ denotes a corrupted pair (e.g., due to annotator error, low effort, adversarial injection, or reward hacking). We model $Z_x \sim \text{Bernoulli}(\rho)$ with unknown base rate $\rho \in (0, 1)$.

**Definition 2 (Verifier Agent).** Let $\mathcal{A} = \{A_1, A_2, A_3, A_4\}$ be a set of $N = 4$ specialized verifier agents, corresponding respectively to the dimensions of **factual correctness** ($A_1$), **safety and policy compliance** ($A_2$), **ethical appropriateness** ($A_3$), and **trustworthiness** ($A_4$). Each agent $A_i$ receives $x$ and applies an internal scoring function followed by a fixed threshold $\tau_i \in (0,1)$ to produce a binary acceptance decision:

$$B_i = \mathbf{1}[S_i \geq \tau_i] \in \{0, 1\},$$

where $S_i \in [0,1]$ is the agent's continuous score. In the theoretical analysis we work directly with the binary decisions $B_i$.

**Definition 3 (Agent Reliability Parameters).** For each agent $A_i$, define the **true positive rate** and **false positive rate** respectively as

$$p_i \triangleq P(B_i = 1 \mid Z_x = 1), \qquad q_i \triangleq P(B_i = 1 \mid Z_x = 0).$$

The quantity $\gamma_i \triangleq p_i - q_i$ is called the **reliability** of agent $A_i$. An agent is called *informative* if $\gamma_i > 0$.

**Assumption 1 (Informative Agents).** All agents are informative: $p_i > q_i$ for all $i \in \{1,2,3,4\}$. This is satisfied when each agent's evaluation dimension is genuinely predictive of pair quality.

**Assumption 2 (Conditional Independence).** Conditioned on $Z_x$, the binary decisions $B_1, B_2, B_3, B_4$ are mutually independent:

$$P(B_1, \ldots, B_N \mid Z_x) = \prod_{i=1}^{N} P(B_i \mid Z_x).$$

This assumption is reasonable when the four agents evaluate genuinely orthogonal dimensions of quality. Its limitations are discussed in Section 4.

---

## 2. The $(k,N)$-Consensus Filter

**Definition 4 (Consensus Filter).** Given agent decisions $B_1, \ldots, B_N$, define the aggregate vote:

$$V(x) \triangleq \sum_{i=1}^{N} B_i \in \{0, 1, \ldots, N\}.$$

The **$(k,N)$-consensus filter** accepts preference pair $x$ if and only if

$$\mathcal{F}_{k,N}(x) = 1 \iff V(x) \geq k,$$

for a chosen consensus threshold $k \in \{1, \ldots, N\}$. In the current implementation with $N=4$ agents, the choice $k=3$ corresponds to a strict majority rule requiring three of the four agents to pass the pair.

**Definition 5 (Filtering Error Probabilities).** Define the two fundamental error probabilities of the filter:

$$\alpha(k, N) \triangleq P\bigl(V(x) \geq k \mid Z_x = 0\bigr) \quad \text{(corruption pass-through)},$$

$$\beta(k, N) \triangleq P\bigl(V(x) < k \mid Z_x = 1\bigr) \quad \text{(valid pair rejection)}.$$

A well-designed filter minimizes $\alpha(k,N)$ (corrupted pairs admitted) while controlling $\beta(k,N)$ (valid pairs discarded), subject to downstream DPO training requirements.

---

## 3. Exact Binomial Analysis (Homogeneous Case)

To derive closed-form expressions, we first analyze the case where all agents share identical reliability parameters.

**Assumption 3 (Homogeneous Agents).** $p_i = p$ and $q_i = q$ for all $i \in [N]$, with $p > q$.

Under Assumptions 2 and 3, the vote counts under each hypothesis are exactly Binomial:

$$V \mid Z=1 \sim \mathrm{Binomial}(N, p), \qquad V \mid Z=0 \sim \mathrm{Binomial}(N, q).$$

**Proposition 1 (Exact Binomial Forms).** Under Assumptions 2 and 3:

$$\alpha(k, N) = \sum_{j=k}^{N} \binom{N}{j} q^j (1-q)^{N-j},$$

$$\beta(k, N) = \sum_{j=0}^{k-1} \binom{N}{j} p^j (1-p)^{N-j}.$$

*Proof.* Since $B_i \mid Z=0 \overset{\mathrm{i.i.d.}}{\sim} \mathrm{Bernoulli}(q)$, their sum $V \mid Z=0 \sim \mathrm{Binomial}(N,q)$ by Assumption 2. The expression for $\alpha(k,N)$ is the upper tail $P(V \geq k)$ of this distribution evaluated at the threshold $k$. The derivation of $\beta(k,N)$ is identical under $Z=1$. $\square$

These expressions can equivalently be written using the regularized incomplete beta function:

$$\alpha(k,N) = I_q(k,\, N - k + 1), \qquad \beta(k,N) = I_{1-p}(N - k + 1,\, k),$$

which enables efficient computation for large $N$.

**Proposition 2 (Monotonicity).** Under Assumptions 2 and 3:

1. $\alpha(k+1, N) < \alpha(k, N)$ for all $k < N$ (increasing $k$ reduces pass-through).
2. $\beta(k, N) < \beta(k-1, N)$ for all $k > 1$ (decreasing $k$ reduces valid rejection).
3. For fixed ratio $\kappa = k/N$ with $q < \kappa < p$: $\alpha(k,N) \to 0$ and $\beta(k,N) \to 0$ as $N \to \infty$.

*Proof sketch.* Parts (1) and (2) follow directly from the fact that the Binomial CDF is strictly monotone in the threshold. Part (3) follows from the Law of Large Numbers: $V/N \xrightarrow{P} q$ under $Z=0$ and $V/N \xrightarrow{P} p$ under $Z=1$, so both tail probabilities vanish as $N \to \infty$ whenever $\kappa$ lies strictly between $q$ and $p$. $\square$

Part (3) establishes that consistent filtering is achievable in the large-agent limit, provided $q < k/N < p$.

---

## 4. Exponential Upper Bound on $\alpha(k,N)$

Proposition 1 gives exact values but limited analytical insight. The following theorem provides a tight exponential upper bound that explicitly reveals the role of the KL divergence between the filter threshold and the agent false positive rate.

**Theorem 1 (Chernoff Bound on Corruption Pass-Through).** Under Assumptions 2 and 3, if $k/N > q$, then:

$$\alpha(k, N) \leq \exp\!\Bigl(-N \cdot \mathrm{KL}\bigl(k/N \,\|\, q\bigr)\Bigr),$$

where $\mathrm{KL}(a \| b) = a \log(a/b) + (1-a)\log\bigl((1-a)/(1-b)\bigr)$ is the binary KL divergence.

*Proof.* Let $X = V \mid Z=0 \sim \mathrm{Binomial}(N, q)$ and let $t = k/N > q$. By the Chernoff method, for any $\lambda > 0$:

$$P(X \geq k) = P\!\left(\frac{X}{N} \geq t\right) \leq e^{-\lambda k} \cdot \mathbb{E}\bigl[e^{\lambda X}\bigr] = e^{-\lambda k} \cdot \bigl(1 - q + q e^{\lambda}\bigr)^N.$$

Taking the infimum over $\lambda > 0$ and evaluating the minimizer $\lambda^* = \log\!\bigl[t(1-q)\big/\bigl(q(1-t)\bigr)\bigr] > 0$ (which is positive since $t > q$) yields the moment-generating function bound:

$$\inf_{\lambda > 0}\; e^{-\lambda k}(1 - q + q e^\lambda)^N = \exp\!\left(-N \cdot \mathrm{KL}(t \| q)\right). \quad \square$$

**Corollary 1 (Agent Count Sufficiency).** To achieve $\alpha(k,N) \leq \delta$ for a target failure probability $\delta \in (0,1)$, it suffices to use

$$N \geq \frac{\log(1/\delta)}{\mathrm{KL}(k/N \,\|\, q)}.$$

This gives a concrete, quantitative justification for the number of agents required to achieve a desired level of corruption robustness.

**Corollary 2 (Exponential Decay).** For fixed $k/N = \kappa$, the corruption pass-through decays exponentially in $N$: $\alpha(k,N) = O\!\bigl(\exp(-cN)\bigr)$ where $c = \mathrm{KL}(\kappa \| q) > 0$.

---

## 5. Limitations: The Correlated Agent Case

Assumption 2 (conditional independence) may not hold in practice. If two agents — for example, the **trustworthiness** agent ($A_4$) and the **factual correctness** agent ($A_1$) — are both implemented using the same underlying language model, their decisions will be positively correlated even after conditioning on $Z_x$.

Let $\rho_{ij}^{(z)} = \mathrm{Corr}(B_i, B_j \mid Z=z)$ denote the residual pairwise correlation under class $z$. When agents are positively correlated under $Z=0$ (i.e., $\rho_{ij}^{(0)} > 0$), the variance of $V \mid Z=0$ is inflated relative to the independent case:

$$\mathrm{Var}(V \mid Z=0) = \sum_i q_i(1-q_i) + \sum_{i \neq j} \rho_{ij}^{(0)} \sqrt{q_i(1-q_i)\,q_j(1-q_j)}\; > \;\sum_i q_i(1-q_i).$$

A Chebyshev-based bound under positive correlation gives:

$$\alpha(k,N) \leq \frac{\mathrm{Var}(V \mid Z=0)}{(k - \bar{q}N)^2},$$

which degrades polynomially in $N$ rather than exponentially. This degradation underscores that **architectural diversity** — using different backbone models and evaluation prompts for each agent — is not merely an engineering convenience but a theoretical necessity for maintaining the exponential filtering guarantee of Theorem 1.

We treat the full correlated analysis as future work, noting that it connects to the literature on correlated voting models (e.g., Nitzan & Paroush, 1982) and dependent crowdsourcing estimators.

---

## 6. Weighted Aggregation: A Bayesian Extension

The hard-threshold consensus rule discards the continuous score $S_i$ once it is binarized. A natural extension retains this information via a weighted log-odds aggregator.

**Definition 6 (Bayesian Score).** Given a prior $\rho = P(Z=1)$ and continuous scores $s_1, \ldots, s_N$, define the log-posterior-odds:

$$\Lambda(x) \triangleq \log\frac{\rho}{1-\rho} + \sum_{i=1}^{N} \lambda_i(s_i),$$

where $\lambda_i(s_i) = \log\bigl[p_i(s_i \mid Z=1) / p_i(s_i \mid Z=0)\bigr]$ is the per-agent log-likelihood ratio. Accept $x$ if $\Lambda(x) \geq 0$ (equivalently, $P(Z=1 \mid s_{1:N}) \geq 0.5$).

Under a tractable linear approximation — valid near the decision boundary — this simplifies to the **reliability-weighted score**:

$$\Lambda_{\mathrm{approx}}(x) = \log\frac{\rho}{1-\rho} + \sum_{i=1}^{N} \gamma_i \cdot (s_i - 0.5),$$

where $\gamma_i = p_i - q_i$ weights each agent's contribution by its reliability.

**Remark (Relationship to Consensus Filter).** The hard-threshold consensus filter is a special case of Definition 6 in which each $s_i$ is replaced by its binarization $B_i$ and all agents are given equal weight. The weighted score $\Lambda_{\mathrm{approx}}$ therefore generalizes the $(k,N)$-filter while remaining implementable within the same codebase by replacing the vote count $V(x)$ with a weighted sum of continuous scores.

**Remark (Neyman-Pearson Optimality).** Under Assumption 2, if the per-agent score densities $p_i(s \mid z)$ are known, Definition 6 yields the **Neyman-Pearson optimal test** for the hypothesis $H_0: Z=0$ vs.\ $H_1: Z=1$ at a given level. This claim requires that (i) conditional independence holds exactly, (ii) the score densities are correctly specified, and (iii) the prior $\rho$ is known. In practice these conditions are at best approximate, and we present this only as a theoretical motivator for the weighted aggregation rule rather than a practical guarantee.

---

## 7. Alignment with Implementation

The theoretical quantities defined above map directly to the existing codebase as follows.

| Theoretical object | Code object | Notes |
|---|---|---|
| $Z_x \in \{0,1\}$ | Latent; unobserved | Estimated via posterior |
| $S_i \in [0,1]$ | `AgentResult.score` | Continuous output per agent |
| $\tau_i$ | `agent.threshold` | Set to 0.7 for all agents |
| $B_i = \mathbf{1}[S_i \geq \tau_i]$ | `AgentResult.passed` | Binary decision |
| $p_i,\, q_i$ | Estimated on calibration data | Used to compute $\gamma_i$ |
| $\gamma_i = p_i - q_i$ | `reliability[i]` | Passed to `MAVFFilter` |
| $V(x) = \sum_i B_i$ | `MAVFResult.vote_count` | Integer in $\{0,1,2,3,4\}$ |
| $\alpha(k,N)$ | `estimate_alpha_beta()` | Empirical on labeled subset |
| $\Lambda(x)$ | `MAVFResult.log_posterior_odds` | Weighted Bayesian score |

The four agents $\{A_1, A_2, A_3, A_4\}$ correspond to `KnowledgeVerifier`, `BehaviorAuditor`, `EthicsEvaluator`, and `TrustAssessor` respectively, each evaluating an orthogonal dimension. Their architectural separation across prompt templates and roles is precisely the practical mechanism by which Assumption 2 is approximated.

---

## Notation Summary

| Symbol | Meaning |
|---|---|
| $x = (q, r_+, r_-)$ | Preference pair |
| $Z_x \in \{0,1\}$ | Latent validity ($1$ = valid, $0$ = corrupted) |
| $\rho = P(Z_x = 1)$ | Base rate of valid pairs |
| $N$ | Number of verifier agents (here $N=4$) |
| $k$ | Consensus threshold |
| $S_i \in [0,1]$ | Continuous score from agent $A_i$ |
| $B_i \in \{0,1\}$ | Binary decision from agent $A_i$ |
| $\tau_i$ | Score threshold for agent $A_i$ |
| $p_i = P(B_i=1 \mid Z=1)$ | True positive rate of $A_i$ |
| $q_i = P(B_i=1 \mid Z=0)$ | False positive rate of $A_i$ |
| $\gamma_i = p_i - q_i$ | Reliability of agent $A_i$ |
| $V(x) = \sum_i B_i$ | Total vote count |
| $\alpha(k,N)$ | Corruption pass-through probability |
| $\beta(k,N)$ | Valid pair rejection probability |
| $\Lambda(x)$ | Log-posterior-odds (Bayesian score) |
| $\mathrm{KL}(a \| b)$ | Binary KL divergence |
| $I_x(a,b)$ | Regularized incomplete beta function |