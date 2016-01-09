import numpy as np
import matplotlib.pyplot as plot

class Network_1D:
    def __init__(self,X,Y,lr,epochs):
        self.num_train = X.shape[0]
        self.x = np.zeros((self.num_train,2))
        self.x[:,0] =X
        self.x[:,1] = 1
        self.y=Y
        self.w = np.asarray([0.0,0.0])
        self.lr = lr
        self.epochs = epochs


    def evaluate_network(self,x,w):
        return np.dot(x,w)

    def cost(self,x,y):
        predict = self.evaluate_network(x,self.w)
        cost = (predict -y) ** 2;
        
        n = x.shape[0] 
        return np.sum(cost)/n

    def cost_gradient(self,x,w,y):
        c = self.evaluate_network(x,w) - y
        gradient = 2 * c * x
        return gradient


    def update_weights(self,gradient):
        self.w = self.w - self.lr*gradient;

    def train(self):
        for t in range(self.epochs):
            gradient = np.zeros_like(self.w)
            for i in range(self.num_train):
                gradient += self.cost_gradient(self.x[i],self.w,self.y[i])
            gradient = gradient/self.num_train
            self.update_weights(gradient)
            cost = self.cost(self.x,self.y)
            print "Epoch %d: w1-%.2f  w2-%.2f  cost-%.2f" % (t,self.w[0],self.w[1],cost)
            #self.plot_output()
                  
    def plot_output(self):
        y = self.w[0]*self.x[:,0] + self.w[1]
        plot.plot(self.x[:,0],y)
        plot.plot(self.x[:,0],self.y,'r.')
        plot.show() 

if __name__ == "__main__":
    x=np.arange(0,6,0.5)
    l=x.shape[0]
    m=10
    c=20
    s= np.random.normal(0.2,0.3,l)
    y=[m*x_ + c +s_ for x_,s_ in zip(x,s)]
    lr = 0.001
    epochs = 5000

    Net = Network_1d(x,y,lr,epochs)
    Net.train()
    Net.plot_output()
    



