"""
n pytorch all that would change once you have the logits is that you would pass it through torch.nn.MultiMargjnLoss() instead if torch.nn.CrossEntropyLoss().
"""
import numpy
import torch
import torch.nn
import torch.optim
import torch.utils
torch.manual_seed(0)

def calculate_hinge_loss(actual, predicted):
    """
    l(y,y_hat) = max(0, 1-y*y_hat) where actual contains -1, 1
    NOTE: Definition makes it look like an inner product
    """
    # numpy.mean(numpy.maximum(0, numpy.subtract(1, numpy.multiply(actual_p, predicted_p))))
    out = torch.mean(torch.clamp(torch.sub(1, torch.mul(actual, predicted)), min = 0))

    return out

def calculate_svm(vector_observed, vector_weights, vector_bias):
    return torch.inner(vector_observed, vector_weights) + vector_bias


lr=0.1, weight_decay=0.01

def train_svm_linear(dataloader, instance_model, num_epochs, **kwargs):

    w = torch.randn(1, 2, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    instance_optimizer = torch.optim.SGD([w,b], **kwargs)

    instance_model.train()
    for epoch in range(num_epochs):
        for (xi, yi) in dataloader: # batch = (xi, yi)
            instance_optimizer.zero_grad() # zero gradients in all variables in this optimizer
            # output = instance_model(xi).squeeze() # removes inputs of size 1
            # weight = instance_model.weight.squeeze() # removes inputs of size 1
            # loss = calculate_hinge_loss(yi, output)
            # loss +=

            output = calculate_svm(xi, w, b)
            loss = calculate_hinge_loss(yi, output.squeeze())
            # loss +=
            loss.backward() # compute gradient for all parameters with require_grad = True
            instance_optimizer.step() # update the gradient parameters

    return w, b

def worker_init_fn(worker_id):                                                          
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)


def visualize(X, Y, model):
    W = model.weight.squeeze().detach().cpu().numpy()
    b = model.bias.squeeze().detach().cpu().numpy()

    delta = 0.001
    x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
    y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
    x, y = np.meshgrid(x, y)
    xy = list(map(np.ravel, [x, y]))

    z = (W.dot(xy) + b).reshape(x.shape)
    z[np.where(z > 1.0)] = 4
    z[np.where((z > 0.0) & (z <= 1.0))] = 3
    z[np.where((z > -1.0) & (z <= 0.0))] = 2
    z[np.where(z <= -1.0)] = 1

    plt.figure(figsize=(10, 10))
    plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
    plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
    plt.contourf(x, y, z, alpha=0.8, cmap="Greys")
    plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10)
    plt.tight_layout()
    plt.show()


# instance_model = torch.nn.Linear(2, 1)
# instance_optimizer = torch.optim.SGD(instance_model.parameters(), lr=0.1, weight_decay=0.01)



print("YES")