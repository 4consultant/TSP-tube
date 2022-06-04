class RowCol:
    def __init__(self, index):
        self.index = index[:, 0:2]
        self.total_path = self.index

    def train(self):
        self.index = self.index[self.index[:, 0].argsort()]
        index1 = 0
        label = 0
        for i in range(1, len(self.index)):
            if self.index[i, 0] != self.index[i - 1, 0] or i == len(self.index) - 1:
                temp = self.index[index1:i]
                temp = temp[temp[:, 1].argsort()]
                if label % 2:
                    temp = temp[::-1]
                self.total_path[index1:i] = temp
                index1 = i
                label += 1
        self.total_path = self.total_path[0:-1]
        return self.total_path

