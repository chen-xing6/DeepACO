\begin{algorithm}[H]
\caption{DeepACO with MCTS for TSP}
\label{alg:deepaco_mcts}
\begin{algorithmic}[1]
\Require Distance matrix $D \in \mathbb{R}^{n \times n}$, number of ants $m$, decay factor $\rho$, learning rate $\eta$, number of epochs $E$, steps per epoch $S$, MCTS simulations $K$
\Ensure Trained neural network $\theta$ for heuristic matrix generation
\State Initialize neural network $\theta$ with parameters for TSP graph encoding
\State Initialize pheromone matrix $P \leftarrow \mathbf{1}_{n \times n}$
\For{epoch $e = 1$ to $E$}
    \For{step $s = 1$ to $S$}
        \State Sample random TSP instance $G$ with $n$ nodes
        \State Encode graph $G$ using neural network: $H \leftarrow f_\theta(G)$
        \State Initialize ACO with $m$ ants, pheromone $P$, heuristic $H$, MCTS enabled with $K$ simulations
        \State Generate ant paths using ACO with MCTS selection
        \State Compute costs $C$ and log-probabilities $L$ from ant paths
        \State Compute baseline $b \leftarrow \mean(C)$
        \State Compute REINFORCE loss: $\mathcal{L} \leftarrow \frac{1}{m} \sum_{i=1}^m (C_i - b) \cdot L_i$
        \State Update $\theta$ using gradient descent: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$
    \EndFor
    \State Validate on held-out dataset and record performance
\EndFor
\State \Return $\theta$
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[H]
\caption{ACO Path Construction with MCTS}
\label{alg:aco_mcts_path}
\begin{algorithmic}[1]
\Require Distance matrix $D$, pheromone $P$, heuristic $H$, number of ants $m$, MCTS simulations $K$
\Ensure Best tour cost $C^*$
\State Initialize $C^* \leftarrow \infty$
\For{each ant $i = 1$ to $m$}
    \State Start from random node $v_0$
    \State Current path $\pi \leftarrow [v_0]$
    \While{$|\pi| < n$}
        \State Use MCTS to select next node $v \notin \pi$ from current path $\pi$
        \State Append $v$ to $\pi$
    \EndWhile
    \State Compute tour cost $C \leftarrow$ cost of $\pi$ (including return to start)
    \State Update $C^* \leftarrow \min(C^*, C)$
\EndFor
\State Update pheromone matrix using elitist strategy
\State \Return $C^*$
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[H]
\caption{MCTS Selection for Next Node}
\label{alg:mcts_selection}
\begin{algorithmic}[1]
\Require Current partial path $\pi$, distance matrix $D$, pheromone $P$, heuristic $H$, simulations $K$
\Ensure Selected next node $v$
\State Initialize root node with state $\pi$
\For{simulation $k = 1$ to $K$}
    \State Traverse tree: select child with UCB until leaf or unexpanded node
    \State Expand: add new child node for untried action
    \State Simulate: complete random tour from current state
    \State Backpropagate: update values along path
\EndFor
\State Select action $v$ with highest visit count from root
\State \Return $v$
\end{algorithmic}
\end{algorithm}</content>
<parameter name="filePath">/root/DeepACO_MCTS_Algorithm.tex