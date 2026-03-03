
# Simple perceptron implementation
class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=25):
        # initialize weights and bias with 0
        self.w1 = 0
        self.w2 = 0
        self.b = 0
        self.lr = learning_rate
        self.epochs = epochs

    # Step function for decision
    def predict(self, x1, x2):
        z = self.w1*x1 + self.w2*x2 + self.b
        if z >= 0:
            return 1
        else:
            return 0

    # Training function
    def train(self, data):

        for epoch in range(self.epochs):

            for x1, x2, target in data:

                # predicted output
                output = self.predict(x1, x2)

                # error calculation
                error = target - output

                # update weights and bias
                self.w1 = self.w1 + self.lr * error * x1
                self.w2 = self.w2 + self.lr * error * x2
                self.b  = self.b  + self.lr * error

        print("Final weights:", self.w1, self.w2)
        print("Final bias:", self.b)

    # Testing function
    def test(self, data):
        print("\nChecking predictions:")
        for x1, x2, target in data:
            result = self.predict(x1, x2)
            print(x1, x2, "-> Predicted:", result, "Actual:", target)

AND_data = [
    (0,0,0),
    (0,1,0),
    (1,0,0),
    (1,1,1)
]

print("Training AND gate")
p = Perceptron()
p.train(AND_data)
p.test(AND_data)

OR_data = [
    (0,0,0),
    (0,1,1),
    (1,0,1),
    (1,1,1)
]

print("\nTraining OR gate")
p = Perceptron()
p.train(OR_data)
p.test(OR_data)

NAND_data = [
    (0,0,1),
    (0,1,1),
    (1,0,1),
    (1,1,0)
]

print("\nTraining NAND gate")
p = Perceptron()
p.train(NAND_data)
p.test(NAND_data)

NOR_data = [
    (0,0,1),
    (0,1,0),
    (1,0,0),
    (1,1,0)
]

print("\nTraining NOR gate")
p = Perceptron()
p.train(NOR_data)
p.test(NOR_data)

XOR_data = [
    (0,0,0),
    (0,1,1),
    (1,0,1),
    (1,1,0)
]

print("\nTraining XOR gate")
p = Perceptron()
p.train(XOR_data)
p.test(XOR_data)

