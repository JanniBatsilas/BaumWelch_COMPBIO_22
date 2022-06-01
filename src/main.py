import matplotlib.pyplot as plt
import numpy as np


class BaumWelch:

    def __init__(self):

        self.tmat = np.array([[0.6, 0.4], [0.4, 0.6]])  # transitions
        self.emat = np.array([[0.2, 0.2, 0.3, 0.3], [0.25, 0.25, 0.25, 0.25]])  # emissions
        self.startmat = np.array([0.5, 0.5])  # start probabilities
        self.atgc = {"A": 0, "T": 1, "G": 2, "C": 3}

        # get Sequences for training from file
        f = open("SequencesCpG.txt", "r")
        counter = 0
        self.sequences = []
        for sequence in f:
            counter = counter + 1
            self.sequences.append(sequence[:-1])  # IGNORE NEW LINE OPERATOR !
            # 7 because these are the ones i have chosen for training
            if counter == 7:
                break

    def forward(self, sequence):

        f = np.zeros((2, len(sequence)))
        # Calculate the start probabilities
        f[0][0] = self.startmat[0] * self.emat[0][self.atgc[sequence[0]]]
        f[1][0] = self.startmat[1] * self.emat[1][self.atgc[sequence[0]]]
        # map every letter to a number according self.atgc
        e_probs = [*map(self.atgc.get, sequence)]

        for i in range(len(sequence)):
            for state in range(2):
                f[state][i] += ((f[0][i - 1] * self.tmat[0][state]) + (f[1][i - 1] * self.tmat[1][state])) * self.emat[state][e_probs[i]]
        return f, (f[0, :] + f[1, :])[-1]

    def backward(self, sequence):

        b = np.zeros((2, len(sequence)))
        # Set the end probabilities to 1
        b[0][-1] = 1
        b[1][-1] = 1
        # map every letter to a number according self.atgc
        e_probs = [*map(self.atgc.get, sequence)]

        for i in np.arange(len(sequence) - 2, -1, -1):
            for state in range(2):
                b[state][i] += b[0][i + 1] * self.tmat[state][0] * self.emat[0][e_probs[i + 1]] + b[1][i + 1] * \
                               self.tmat[state][1] * self.emat[1][e_probs[i + 1]]
        return b

    def run(self, eps):

        while (True):

            A = np.zeros((2, 2))
            E = np.zeros((2, 4))
            sum_log_p = 0

            # Iterating over sequences (SLIDE 64 LECTURE 4)
            for seq in self.sequences:
                f, P = self.forward(seq)
                b = self.backward(seq)
                sum_log_p += np.log(P)

                # Iterating over all positions
                for i in range(len(seq) - 1):

                    E[0][self.atgc[seq[i]]] = E[0][self.atgc[seq[i]]] + f[0][i] * b[0][i] / P
                    E[1][self.atgc[seq[i]]] = E[1][self.atgc[seq[i]]] + f[1][i] * b[1][i] / P

                    for a in range(2):
                        A[a, 0] += (1/P) * f[a][i] * self.tmat[a][0] * self.emat[0][self.atgc[seq[i + 1]]] * b[0][i + 1]
                        A[a, 1] += (1/P) * f[a][i] * self.tmat[a][1] * self.emat[1][self.atgc[seq[i + 1]]] * b[1][i + 1]

            self.tmat[0] = A[0] / sum(A[0])
            self.emat[0] = E[0] / sum(E[0])
            self.tmat[1] = A[1] / sum(A[1])
            self.emat[1] = E[1] / sum(E[1])

            diff = 0.
            for i in range(len(self.sequences)):
                f, P = self.forward(self.sequences[i])
                diff += np.log(P)

            if abs(sum_log_p - diff) < eps:
                break

    def calculate_pi_k_given_x(self, f, b, P):
        return (np.multiply(f, b) / P)[0]


    # TODO: SOMEHOW TWO PLOT WINDOWS OPEN, I DO NOT KNOW WHY :(
    def print_results(self, f, b, P):

        y = baum.calculate_pi_k_given_x(f, b, P)
        plt.title("Baum-Welch CpG-probability-Detection ")
        plt.plot(y, label="P(CpG)", color='red')
        plt.xlabel('index of base')
        plt.ylabel('P(CpG)')

        plt.legend(loc='upper right')
        plt.figure(figsize=(10, 5))
        plt.show()


if __name__ == '__main__':
    cpg_seq = "TTCCTCTCTCTCTCCGCGCGCGCGAAAGCGCGTCTCTCTCTCTCTACTTGAATAAATATTGTCGGGGGCGCGAAGCGTCGCTCACTTATTCCGACCGGATGATCTCCAATAAAAAAGTATAATTATATATCATATTAATATATTCATGTCATCTAGGATAAATATATAAGATATATAATTTAATTAAGAATATTTTATACTACTATATATTT"

    baum = BaumWelch()
    baum.run(0.001)
    f, P = baum.forward(cpg_seq)
    b = baum.backward(cpg_seq)
    baum.print_results(f, b, P)
