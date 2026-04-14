def create_a_network():
    tot_neuron=10
    num_edge_from_neuron=1
    num_edge_to_neuron=1
    num_edge=tot_neuron*num_edge_from_neuron
    Weight=np.random.rand(num_edge,)
    Delay=np.zeros(num_edge,)

    From=np.zeros(tot_neuron*num_edge_from_neuron,dtype="int")
    To=np.zeros(tot_neuron*num_edge_from_neuron,dtype="int")
    for epn in range(num_edge_from_neuron):
        From[epn*tot_neuron:(epn+1)*tot_neuron]=np.arange(tot_neuron)
    for epn in range(num_edge_to_neuron):
        To[epn*tot_neuron:(epn+1)*tot_neuron]=np.arange(tot_neuron)

    np.random.shuffle(From)
    np.random.shuffle(To)
    
    return From,To,Weight,Delay

class Genetic_Algorithm:
    
    def __init__(self,network):
        self.crossover_rate=0.1
        self.network=network
        self.L=len(network)
        
    def selection(self):
        obj=[self.objective(net) for net in self.network]
        self.parent=[self.network[i] for i in np.argsort(np.array(obj))[int(self.L//2):]]
        
    def crossover(self,net1,net2): 
        return self.mutation([np.concatenate((net1[i][:int(self.L*0.5)],net2[i][int(self.L*0.5):])) for i in range(4)]),self.mutation([np.concatenate((net1[i][int(self.L*0.5):],net2[i][:int(self.L*0.5)])) for i in range(4)])
    
    def create_new_pop_by_crossover_mutation(self):
        self.new_network=[]
        for i in range(len(self.parent)//2):
            children=self.crossover(self.parent[i*2],self.parent[i*2+1])
            self.new_network.append(self.parent[i*2])
            self.new_network.append(self.parent[i*2+1])
            self.new_network.append(children[0])
            self.new_network.append(children[0])
        
    def mutation(self,net1):
        net1[2][net1[2]<0.1]+=0.1
        net1[2][net1[2]>0.9]-=0.1 
        return net1
    
    def objective(self,net):
        return sum(net[2])