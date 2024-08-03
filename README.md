# Algorithms
Here is a list of common and advanced algorithms with an example use for each. The examples alternate between different programming languages.

### **1. Quantum Gates and Operations**

#### **Hadamard Gate (H)**
- **Description:** Creates a superposition state from the basis states.
- **Example Use:** Quantum superposition in creating an equal probability distribution of qubit states.
- **Example Code (Qiskit):**
  ```python
  from qiskit import QuantumCircuit, Aer, execute
  
  circuit = QuantumCircuit(1)
  circuit.h(0)
  circuit.measure_all()
  
  simulator = Aer.get_backend('qasm_simulator')
  result = execute(circuit, simulator).result()
  print(result.get_counts())
  ```

#### **Pauli-X Gate (X)**
- **Description:** Flips the state of a qubit (similar to a classical NOT gate).
- **Example Use:** Bit-flip operations in quantum algorithms.
- **Example Code (Cirq):**
  ```python
  import cirq
  
  qubit = cirq.LineQubit(0)
  circuit = cirq.Circuit(cirq.X(qubit))
  print(circuit)
  ```

#### **CNOT Gate (Controlled-NOT)**
- **Description:** Conditional bit-flip depending on the state of the control qubit.
- **Example Use:** Entangling two qubits.
- **Example Code (PyQuil):**
  ```python
  from pyquil import Program
  from pyquil.gates import CNOT
  
  p = Program()
  p += CNOT(0, 1)
  print(p)
  ```

#### **Toffoli Gate (Controlled-Controlled-NOT)**
- **Description:** Conditional flip of the target qubit if both control qubits are in state |1⟩.
- **Example Use:** Error correction and quantum computing operations.
- **Example Code (Qiskit):**
  ```python
  from qiskit import QuantumCircuit
  
  qc = QuantumCircuit(3)
  qc.ccx(0, 1, 2)
  qc.measure_all()
  print(qc)
  ```

#### **Swap Gate**
- **Description:** Exchanges the states of two qubits.
- **Example Use:** Reordering qubits in quantum algorithms.
- **Example Code (Cirq):**
  ```python
  import cirq
  
  qubits = cirq.LineQubit.range(2)
  circuit = cirq.Circuit(cirq.SWAP(qubits[0], qubits[1]))
  print(circuit)
  ```

#### **S Gate (Phase Gate)**
- **Description:** Applies a phase of π/2 to the qubit state.
- **Example Use:** Phase manipulation in quantum computing.
- **Example Code (Qiskit):**
  ```python
  from qiskit import QuantumCircuit
  
  qc = QuantumCircuit(1)
  qc.s(0)
  qc.measure_all()
  print(qc)
  ```

#### **T Gate (π/8 Gate)**
- **Description:** Applies a phase of π/4 to the qubit state.
- **Example Use:** Precision phase control in quantum computations.
- **Example Code (Cirq):**
  ```python
  import cirq
  
  qubit = cirq.LineQubit(0)
  circuit = cirq.Circuit(cirq.T(qubit))
  print(circuit)
  ```

### **2. Classical Algorithms**

#### **Bubble Sort**
- **Description:** Simple sorting algorithm that repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order.
- **Example Use:** Sorting small datasets.
- **Example Code (Python):**
  ```python
  def bubble_sort(arr):
      n = len(arr)
      for i in range(n):
          for j in range(0, n-i-1):
              if arr[j] > arr[j+1]:
                  arr[j], arr[j+1] = arr[j+1], arr[j]
      return arr
  
  print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))
  ```

#### **Quick Sort**
- **Description:** Efficient sorting algorithm that uses divide-and-conquer strategy to sort elements.
- **Example Use:** Sorting large datasets efficiently.
- **Example Code (Java):**
  ```java
  public class QuickSort {
      public static void quickSort(int[] arr, int low, int high) {
          if (low < high) {
              int pi = partition(arr, low, high);
              quickSort(arr, low, pi - 1);
              quickSort(arr, pi + 1, high);
          }
      }
      
      private static int partition(int[] arr, int low, int high) {
          int pivot = arr[high];
          int i = (low - 1);
          for (int j = low; j < high; j++) {
              if (arr[j] <= pivot) {
                  i++;
                  int temp = arr[i];
                  arr[i] = arr[j];
                  arr[j] = temp;
              }
          }
          int temp = arr[i + 1];
          arr[i + 1] = arr[high];
          arr[high] = temp;
          return i + 1;
      }
      
      public static void main(String[] args) {
          int[] arr = {10, 7, 8, 9, 1, 5};
          quickSort(arr, 0, arr.length - 1);
          for (int i : arr) {
              System.out.print(i + " ");
          }
      }
  }
  ```

#### **Dijkstra’s Algorithm**
- **Description:** Computes the shortest paths from a source vertex to all other vertices in a weighted graph.
- **Example Use:** Pathfinding in network routing.
- **Example Code (Python):**
  ```python
  import heapq
  
  def dijkstra(graph, start):
      heap = [(0, start)]
      distances = {vertex: float('infinity') for vertex in graph}
      distances[start] = 0
      while heap:
          (cost, u) = heapq.heappop(heap)
          for neighbor, weight in graph[u].items():
              distance = cost + weight
              if distance < distances[neighbor]:
                  distances[neighbor] = distance
                  heapq.heappush(heap, (distance, neighbor))
      return distances
  
  graph = {
      'A': {'B': 1, 'C': 4},
      'B': {'A': 1, 'C': 2, 'D': 5},
      'C': {'A': 4, 'B': 2, 'D': 1},
      'D': {'B': 5, 'C': 1}
  }
  print(dijkstra(graph, 'A'))
  ```

#### **RSA Algorithm**
- **Description:** Public-key cryptographic algorithm for secure data transmission.
- **Example Use:** Secure communication over the internet.
- **Example Code (Python with `cryptography` library):**
  ```python
  from cryptography.hazmat.primitives.asymmetric import rsa
  from cryptography.hazmat.primitives import serialization
  
  # Generate RSA keys
  private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
  public_key = private_key.public_key()
  
  # Serialize private key
  private_pem = private_key.private_bytes(
      encoding=serialization.Encoding.PEM,
      format=serialization.PrivateFormat.TraditionalOpenSSL,
      encryption_algorithm=serialization.NoEncryption()
  )
  
  # Serialize public key
  public_pem = public_key.public_bytes(
      encoding=serialization.Encoding.PEM,
      format=serialization.PublicFormat.SubjectPublicKeyInfo
  )
  
  print("Private Key:", private_pem.decode('utf-8'))
  print("Public Key:", public_pem.decode('utf-8'))
  ```

#### **Linear Regression**
- **Description:** A statistical method to model the relationship between a dependent variable and one or more independent variables.
- **Example Use:** Predicting sales based on advertising spend.
- **Example Code (Python with `scikit-learn`):**
  ```python
  from sklearn.linear_model import LinearRegression
  import numpy as np
  
  # Sample data
  X = np.array([[1], [2], [3], [4], [5]])
  y = np.array([1, 2, 1.3, 3.75, 2.25])
  
  # Create and fit model
  model = LinearRegression()
  model.fit(X, y)
  
  # Make predictions
  predictions = model.predict(np.array([[6]]))
  print(predictions)
  ```

#### **Matrix Chain Multiplicatation**
The description, example use, and provided Python code for the Matrix Chain Multiplication problem are almost correct, but there is a minor mistake in the code. Here's a corrected version of the code:

- **Description:** An optimization problem to determine the most efficient way to multiply a chain of matrices.
- **Example Use:** Reducing computational cost in matrix multiplication.
- **Example Code (Python):**
  ```python
  import numpy as np
  
  def matrix_chain_order(p):
      n = len(p) - 1
      m = [[0] * n for _ in range(n)]
      for l in range(2, n + 1):
          for i in range(n - l + 1):
              j = i + l - 1
              m[i][j] = float('inf')
              for k in range(i, j):
                  q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                  if q < m[i][j]:
                      m[i][j] = q  # Fixed line here
      return m[0][n-1]

  # Example dimensions
  dimensions = [10, 20, 30, 40, 30]
  print("Minimum number of multiplications:", matrix_chain_order(dimensions))
  ```

#### **K-Means Clustering**
- **Description:** A clustering algorithm that partitions data into K distinct clusters based on feature similarity.
- **Example Use:** Customer segmentation in marketing.
- **Example Code (Python with `scikit-learn`):**
  ```python
  from sklearn.cluster import KMeans
  import numpy as np
  
  # Sample data
  X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
  
  # Create and fit the model
  kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
  
  # Get cluster centroids and labels
  centroids = kmeans.cluster_centers_
  labels = kmeans.labels_
  
  print("Centroids:", centroids)
  print("Labels:", labels)
  ```

#### **PageRank Algorithm**
- **Description:** An algorithm used by Google Search to rank web pages in its search engine results.
- **Example Use:** Ranking web pages or nodes in a graph.
- **Example Code (Python):**
  ```python
  import numpy as np
  
  def page_rank(M, num_iterations: int = 100, d: float = 0.85):
      N = M.shape[1]
      v = np.random.rand(N, 1)
      v = v / np.linalg.norm(v, 1)
      M = d * M + (1 - d) / N
      for _ in range(num_iterations):
          v = M @ v
      return v
  
  # Example matrix (transition probabilities)
  M = np.array([[0, 0, 1],
                [1/2, 0, 1/2],
                [1/2, 1, 1/2]])
  
  print("PageRank:", page_rank(M))
  ```

#### **A* Search Algorithm**
- **Description:** An informed search algorithm that finds the shortest path between nodes using heuristics.
- **Example Use:** Pathfinding in games and robotics.
- **Example Code (Python):**
  ```python
  import heapq
  
  def a_star_search(start, goal, graph, heuristic):
      open_set = []
      heapq.heappush(open_set, (0 + heuristic[start], start))
      came_from = {}
      cost_so_far = {start: 0}
      
      while open_set:
          current = heapq.heappop(open_set)[1]
          
          if current == goal:
              break
          
          for neighbor, cost in graph[current].items():
              new_cost = cost_so_far[current] + cost
              if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                  cost_so_far[neighbor] = new_cost
                  priority = new_cost + heuristic[neighbor]
                  heapq.heappush(open_set, (priority, neighbor))
                  came_from[neighbor] = current
                  
      return came_from, cost_so_far
  
  # Example graph and heuristic
  graph = {
      'A': {'B': 1, 'C': 4},
      'B': {'A': 1, 'C': 2, 'D': 5},
      'C': {'A': 4, 'B': 2, 'D': 1},
      'D': {'B': 5, 'C': 1}
  }
  heuristic = {'A': 7, 'B': 6, 'C': 2, 'D': 0}
  
  came_from, cost_so_far = a_star_search('A', 'D', graph, heuristic)
  print("Path:", came_from)
  print("Cost:", cost_so_far)
  ```

#### **Sieve of Eratosthenes**
- **Description:** An efficient algorithm to find all prime numbers up to a specified integer.
- **Example Use:** Finding prime numbers within a range.
- **Example Code (Python):**
  ```python
  def sieve_of_eratosthenes(n):
      primes = [True] * (n + 1)
      p = 2
      while (p * p <= n):
          if primes[p]:
              for i in range(p * p, n + 1, p):
                  primes[i] = False
          p += 1
      return [p for p in range(2, n + 1) if primes[p]]
  
  print("Primes up to 30:", sieve_of_eratosthenes(30))
  ```

#### **Depth-First Search (DFS)**
- **Description:** An algorithm for traversing or searching tree or graph data structures.
- **Example Use:** Solving puzzles, pathfinding.
- **Example Code (Java):**
  ```java
  import java.util.*;

  public class DFS {
      private Map<Integer, List<Integer>> graph = new HashMap<>();
  
      public void addEdge(int v, int w) {
          graph.computeIfAbsent(v, k -> new ArrayList<>()).add(w);
      }
  
      public void dfs(int start) {
          Set<Integer> visited = new HashSet<>();
          dfsUtil(start, visited);
      }
  
      private void dfsUtil(int v, Set<Integer> visited) {
          visited.add(v);
          System.out.print(v + " ");
  
          for (int neighbor : graph.getOrDefault(v, Collections.emptyList())) {
              if (!visited.contains(neighbor)) {
                  dfsUtil(neighbor, visited);
              }
          }
      }
  
      public static void main(String[] args) {
          DFS dfs = new DFS();
          dfs.addEdge(1, 2);
          dfs.addEdge(1, 3);
          dfs.addEdge(2, 4);
          dfs.addEdge(3, 5);
  
          System.out.println("DFS traversal starting from vertex 1:");
          dfs.dfs(1);
      }
  }
  ```

#### **Breadth-First Search (BFS)**
- **Description:** An algorithm for traversing or searching tree or graph data structures level by level.
- **Example Use:** Finding shortest paths in unweighted graphs.
- **Example Code (Python):**
  ```python
  from collections import deque
  
  def bfs(graph, start):
      visited = set()
      queue = deque([start])
      while queue:
          vertex = queue.popleft()
          if vertex not in visited:
              visited.add(vertex)
              print(vertex, end=' ')
              queue.extend(neighbor for neighbor in graph[vertex] if neighbor not in visited)
  
  # Example graph
  graph = {
      1: [2, 3],
      2: [4, 5],
      3: [6, 7],
      4: [],
      5: [],
      6: [],
      7: []
  }
  
  print("BFS traversal starting from vertex 1:")
  bfs(graph, 1)
  ```

#### **Aho-Corasick Algorithm**
- **Description:** An algorithm for searching multiple patterns in a text in linear time.
- **Example Use:** Text search applications like keyword detection.
- **Example Code (Python):**
  ```python
  from ahocorapy.keywordtree import KeywordTree
  
  def aho_corasick_search(text, keywords):
      tree = KeywordTree()
      for keyword in keywords:
          tree.add(keyword)
      tree.finalize()
      return list(tree.search_all(text))
  
  keywords = ['he', 'she', 'his', 'hers']
  text = 'ushers'
  print("Matches found:", aho_corasick_search(text, keywords))
  ```

### **Quantum Computing Algorithms**

#### **Grover’s Algorithm**
- **Description:** Searches an unsorted database or solves an unstructured search problem in √N time.
- **Example Use:** Database searching and optimization problems.
- **Example Code (Qiskit):**
  ```python
  from qiskit import QuantumCircuit, Aer, execute
  from qiskit.algorithms import Grover
  from qiskit.circuit.library import GroverOperator
  from qiskit.circuit import QuantumCircuit
  
  # Define the oracle
  def oracle_circuit():
      qc = QuantumCircuit(2)
      qc.cz(0, 1)
      return qc
  
  # Grover's Algorithm
  grover = Grover(oracle=oracle_circuit(), grover_operator=GroverOperator(oracle_circuit()))
  result = grover.run()
  print("Grover's result:", result)
  ```

#### **Shor’s Algorithm**
- **Description:** Factorizes integers into prime factors using quantum computation.
- **Example Use:** Cryptography and integer factorization.
- **Example Code (Qiskit):**
  ```python
  from qiskit import QuantumCircuit, Aer, execute
  from qiskit.algorithms import Shor

  # Shor's Algorithm
  shor = Shor(15)
  result = shor.run()
  print("Shor's result:", result)
  ```

### **Classical Algorithms and Concepts**

#### **1. Network Flow Algorithms**

- **Ford-Fulkerson Algorithm (Python)**

```python
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(dict)
        self.V = vertices

    def add_edge(self, u, v, w):
        self.graph[u][v] = w

    def bfs(self, s, t, parent):
        visited = [False] * (self.V + 1)
        queue = [s]
        visited[s] = True
        while queue:
            u = queue.pop(0)
            for v, capacity in self.graph[u].items():
                if not visited[v] and capacity > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u
                    if v == t:
                        return True
        return False

    def ford_fulkerson(self, source, sink):
        parent = [-1] * (self.V + 1)
        max_flow = 0
        while self.bfs(source, sink, parent):
            path_flow = float('Inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]
            max_flow += path_flow
        return max_flow

# Example usage
g = Graph(6)
g.add_edge(0, 1, 16)
g.add_edge(0, 2, 13)
g.add_edge(1, 2, 10)
g.add_edge(1, 3, 12)
g.add_edge(2, 1, 4)
g.add_edge(2, 4, 14)
g.add_edge(3, 4, 9)
g.add_edge(3, 5, 20)
g.add_edge(4, 5, 7)

print("Max Flow:", g.ford_fulkerson(0, 5))
```

#### **2. Randomized Algorithms**

- **Monte Carlo Method (Python)**

```python
import random
import math

def estimate_pi(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return (inside_circle / num_samples) * 4

# Example usage
num_samples = 1000000
print("Estimated π:", estimate_pi(num_samples))
```

- **Las Vegas Algorithm (Python)**

```python
import random

def randomized_quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = random.choice(arr)
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    return randomized_quicksort(less) + equal + randomized_quicksort(greater)

# Example usage
arr = [3, 6, 8, 10, 1, 2, 1]
print("Sorted array:", randomized_quicksort(arr))
```

#### **3. Approximation Algorithms**

- **Greedy Algorithm for Fractional Knapsack (Python)**

```python
def fractional_knapsack(weights, values, capacity):
    items = sorted([(v/w, w, v) for w, v in zip(weights, values)], reverse=True)
    total_value = 0
    for ratio, weight, value in items:
        if capacity > 0:
            take_weight = min(weight, capacity)
            total_value += take_weight * ratio
            capacity -= take_weight
    return total_value

# Example usage
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50
print("Maximum value in knapsack:", fractional_knapsack(weights, values, capacity))
```

#### **4. Advanced Graph Algorithms**

- **Tarjan’s Algorithm (Python)**

```python
def tarjan_scc(graph):
    index = 0
    stack = []
    indices = {}
    lowlinks = {}
    on_stack = set()
    sccs = []

    def strongconnect(node):
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] == indices[node]:
            scc = []
            while stack:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == node:
                    break
            sccs.append(scc)

    for v in graph:
        if v not in indices:
            strongconnect(v)

    return sccs

# Example usage
graph = {0: [1], 1: [2], 2: [0, 3], 3: [4], 4: [5], 5: [3]}
print("Strongly Connected Components:", tarjan_scc(graph))
```

- **A* Search Algorithm (Python)**

```python
import heapq

def a_star(start, goal, h):
    open_set = []
    heapq.heappush(open_set, (0 + h[start], start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: h[start]}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor, cost in graph.get(current, []):
            tentative_g_score = g_score[current] + cost
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + h[neighbor]
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

# Example usage
graph = {0: [(1, 1), (2, 4)], 1: [(2, 2), (3, 5)], 2: [(3, 1)], 3: []}
h = {0: 7, 1: 6, 2: 2, 3: 0}
print("Path:", a_star(0, 3, h))
```

### **Quantum Algorithms and Concepts**

#### **1. Quantum Information Theory**

- **Quantum Entanglement (Pseudo-code)**

```pseudo
// Entanglement can be simulated using a quantum programming library like Qiskit.

import qiskit
from qiskit import QuantumCircuit, execute, Aer

# Create a Bell pair (entangled state)
qc = QuantumCircuit(2)
qc.h(0)  # Apply Hadamard gate
qc.cx(0, 1)  # Apply CNOT gate

# Measure the qubits
qc.measure_all()

# Execute the circuit
backend = Aer.get_backend('qasm_simulator')
result = execute(qc, backend, shots=1000).result()
counts = result.get_counts()

print("Measurement results:", counts)
```

#### **2. Quantum Algorithms for Linear Algebra**

- **Quantum PCA (Pseudo-code)**

```pseudo
// Quantum PCA requires a quantum computing framework and is advanced. The following is a high-level description.

import qiskit
from qiskit import QuantumCircuit, Aer, execute

# Create a quantum circuit for PCA
qc = QuantumCircuit(num_qubits)

# Apply quantum operations and gates for PCA
# ...

# Execute the quantum circuit
backend = Aer.get_backend('statevector_simulator')
result = execute(qc, backend).result()
statevector = result.get_statevector()

print("Quantum PCA result:", statevector)
```

#### **3. Quantum Cryptography**

- **Quantum Key Distribution (QKD) (Pseudo-code)**

```pseudo
// Quantum Key Distribution involves quantum communication protocols and can be simulated.

import qiskit
from qiskit import QuantumCircuit, Aer, execute

# Create a quantum circuit for QKD
qc = QuantumCircuit(num_qubits)

# Apply quantum gates for key distribution
# ...

# Execute the circuit
backend = Aer.get_backend('qasm_simulator')
result = execute(qc, backend).result()
key = result.get_counts()

print("Distributed quantum key:", key)
```

For practical implementations of quantum algorithms, specialized quantum programming environments like Qiskit (IBM), Cirq (Google), or QuTiP (Quantum Toolbox in Python) are used. These snippets provide a starting point, but real-world applications may involve more complex setups and quantum hardware integrations. 
