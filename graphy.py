import matplotlib.pyplot as plt

def plot(history_dict, keys, title=None, xyLabel=[], ylim=(), size=()):
    lineType = ('-', '--', '.', ':')    
    if len(ylim)==2: plt.ylim(*ylim)    
    if len(size)==2: plt.gcf().set_size_inches(*size)  
    epochs = range(1, len(history_dict[keys[0]])+1) 
    for i in range(len(keys)):   
        plt.plot(epochs, history_dict[keys[i]], lineType[i])  
    if title:   
        plt.title(title)
    if len(xyLabel)==2:  
        plt.xlabel(xyLabel[0])
        plt.ylabel(xyLabel[1])
    plt.legend(keys, loc='best') 
    plt.show()  
