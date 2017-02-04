import caffe
import numpy as np

class DiceLoss(caffe.Layer):
    """
    Compute energy based on dice coefficient.
    """
    union = None
    intersection = None
    result = None
    gt = None

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute the dice. the result of the softmax and the ground truth.")


    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            print bottom[0].data.shape
            print bottom[1].data.shape
            raise Exception("the dimension of inputs should match")

        # loss output is two scalars (mean and std)
        top[0].reshape(1)


    def forward(self, bottom, top):

        dice = np.zeros((bottom[0].data.shape[0],bottom[0].data.shape[1]),dtype=np.float32)
        print bottom[0].data.shape[0]
        self.union = np.zeros((bottom[0].data.shape[0],bottom[0].data.shape[1]),dtype=np.float32)
        self.intersection = np.zeros((bottom[0].data.shape[0],bottom[0].data.shape[1]),dtype=np.float32)

        self.result = np.reshape(np.squeeze(np.argmax(bottom[0].data[...],axis=1)),
                                 [bottom[0].data.shape[0],bottom[0].data.shape[2]])
        self.gt = np.reshape(np.squeeze(np.argmax(bottom[1].data[...], axis=1)),
                                 [bottom[1].data.shape[0], bottom[1].data.shape[2]])

        self.gt = self.gt.astype(dtype=np.float32)
        self.result = self.result.astype(dtype=np.float32)

        for i in range(0,bottom[0].data.shape[0]):
            for j in range(0,bottom[0].data.shape[1]):
                # compute dice
                CurrResult = (self.result[i,:]==j).astype(dtype=np.float32)
                CurrGT = (self.gt[i,:]==j).astype(dtype=np.float32)
                self.union[i,j]=(np.sum(CurrResult) + np.sum(CurrGT))+0.0001
                self.intersection[i,j]=(np.sum(CurrResult * CurrGT))
                dice[i,j] = 2 * (self.intersection[i,j]) / (self.union[i,j])
            print "dice=",dice[i]
        top[0].data[0]=np.sum(dice)


    def backward(self, top, propagate_down, bottom):
        for btm in [0]:
            prob = bottom[0].data[...]
            bottom[btm].diff[...] = np.zeros(bottom[btm].diff.shape, dtype=np.float32)
            for i in range(0, bottom[btm].diff.shape[0]):
                for j in range(0, bottom[0].data.shape[1]):
                    if self.union[i,j] <= 10:
                        continue
                    bottom[btm].diff[i, j, :] -= 2.0 * (
                        (self.gt[i,j] * self.union[i,j]) / ((self.union[i,j]) ** 2) - 2.0*prob[i,j]*(self.intersection[i,j]) /
                        ((self.union[i,j]) ** 2))

