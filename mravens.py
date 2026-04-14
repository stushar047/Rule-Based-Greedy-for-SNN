import numpy as np

class Neuron:
    def __init__(self):
        self.node_id='0'
        self.type='input'
        self.threshold=0.4
        self.leak=0.0
        self.refr_leak=0.0
        self.refr_potential=-0.0
        self.std_potential=0.0
        self.abs_period=2
        self.charge=0.0
        self.last_fire=-1
        self.fire_count=0
        self.abs_time=1
        self.From = []
        self.To = []
        self.period_type="std"
        self.stdp=False

class Synapse:
    def __init__(self):
        self.weight=0.1 
        self.delay=0
        self.From=0
        self.To=1
        self.pot_pointer=0
        self.dep_pointer=0
        
class create_custom_network:
    
    def __init__(self):
        self.neuron=[]
        self.Syn=[]
        self.s_i=0
        self.tot_neuron=0
        self.num_input_neuron=0
        self.num_output_neuron=0
        
    def add_neuron(self,node_id,n_type):
        n=Neuron()
        n.node_id=node_id
        n.type=n_type
        self.neuron.append(n)
        
        self.tot_neuron+=1
        if n_type=="input":
            self.num_input_neuron+=1   
        if n_type=="output":
            self.num_output_neuron+=1    
        
    def add_synapse(self,From,To,Weight):
        s=Synapse()
        s.From=From
        s.To=To
        s.weight=Weight
        self.Syn.append(s)
        for n in self.neuron:
            if n.node_id==From:
                n.To.append((self.s_i,To,Weight,0))
            if n.node_id==To:    
                n.From.append((self.s_i,From,Weight,0))
        self.s_i+=1        
                
class create_random_network:
    
    def __init__(self,num_input_neuron,num_output_neuron,num_edge_from_neuron,num_edge_to_neuron):
        self.num_input_neuron=num_input_neuron
        self.num_output_neuron=num_output_neuron
        self.num_edge_from_neuron=num_edge_from_neuron
        self.num_edge_to_neuron=num_edge_to_neuron
        self.neuron=[]
        self.Syn=[]
        
    def create_neuron(self,seed=4):
        np.random.seed(seed)
        self.tot_neuron=self.num_input_neuron+self.num_output_neuron
        self.num_edge=self.tot_neuron*self.num_edge_from_neuron
        self.Weight=np.round(np.random.rand(self.num_edge,),2)
        self.Delay=np.zeros(self.num_edge,)
        self.From=np.zeros(self.tot_neuron*self.num_edge_from_neuron,dtype="int")
        self.To=np.zeros(self.tot_neuron*self.num_edge_from_neuron,dtype="int")
        for epn in range(self.num_edge_from_neuron):
            self.From[epn*self.tot_neuron:(epn+1)*self.tot_neuron]=np.arange(self.tot_neuron)
        for epn in range(self.num_edge_to_neuron):
            self.To[epn*self.tot_neuron:(epn+1)*self.tot_neuron]=np.arange(self.tot_neuron)
        
        np.random.shuffle(self.From)
        np.random.shuffle(self.To)
        
        for i in range(self.num_input_neuron):
            n=Neuron()
            n.node_id=i
            n.type="input"
            ss_to=np.where(self.From==i+1)[0]
            ss_from=np.where(self.To==i+1)[0]
            to_n,to_w,to_delay=self.To[ss_to],self.Weight[ss_to],self.Delay[ss_to]
            from_n,from_w,from_delay=self.From[ss_from],self.Weight[ss_from],self.Delay[ss_from]
            n.To=[(i,j,k,l) for (i,j,k,l) in zip(ss_to,to_n,to_w,to_delay)]
            n.From=[(i,j,k,l) for (i,j,k,l) in zip(ss_from,from_n,from_w,from_delay)]
            self.neuron.append(n)
            
        for i in range(self.num_input_neuron,self.tot_neuron):
            n=Neuron()
            n.node_id=i
            n.type="output"
            ss_to=np.where(self.From==i+1)[0]
            ss_from=np.where(self.To==i+1)[0]
            to_n,to_w,to_delay=self.To[ss_to],self.Weight[ss_to],self.Delay[ss_to]
            from_n,from_w,from_delay=self.From[ss_from],self.Weight[ss_from],self.Delay[ss_from]
            n.To=[(i,j,k,l) for (i,j,k,l) in zip(ss_to,to_n,to_w,to_delay)]
            n.From=[(i,j,k,l) for (i,j,k,l) in zip(ss_from,from_n,from_w,from_delay)]
            self.neuron.append(n) 
            
    def create_synapse(self):
        for i in range(len(self.From)):
            s=Synapse()
            s.From=self.From[i]
            s.To=self.To[i]
            s.Weight=self.Weight[i]
            self.Syn.append(s)

class create_spikes:
    def __init__(self,sim_time,net):
        self.events=np.zeros((sim_time,net.tot_neuron))
    def add_event(self,spike_time,neuron,spike_value):
        self.events[spike_time,neuron]=spike_value

class MRAVENS:
    def __init__(self,min_weight,max_weight):
        self.min_weight=min_weight
        self.max_weight=max_weight
        self.stdp_potentiation=[0.1, 0.2, 0.3, 0.4] 
        self.stdp_depression=[0.1, 0.2, 0.3, 0.4]   

class process_event:
    def __init__(self,sime_time,net,events,proc):
        self.sim_time=sime_time
        self.net=net
        self.num_input_neuron=net.num_input_neuron
        self.num_output_neuron=net.num_output_neuron
        self.events=events
        self.proc=proc
        self.potential=np.zeros((self.sim_time,self.num_input_neuron+self.num_output_neuron))
        self.spikes_=np.zeros((self.sim_time,self.num_input_neuron+self.num_output_neuron))
        
    def apply_spike(self):
#         print("nothing")
        #for all the spike events 
        for i in range(self.sim_time):
            print(f"Time:{i}")
            #for all the neurons
            for k in range(self.num_input_neuron+self.net.num_output_neuron):
#                 print(self.net.num_output_neuron)
                #For std period
                print(f"Event at time {i}, neuron {self.net.neuron[k].node_id}: {self.events[i][k]}")
        
                if self.net.neuron[k].period_type=="std":
                    print(f"The neuron {self.net.neuron[k].node_id} is in std in time {i}. So the charge has been changed from {self.net.neuron[k].charge}")
                    #Step 1: charge add and leak 
                    self.net.neuron[k].charge = self.net.neuron[k].charge + self.events[i][k]-self.net.neuron[k].leak
                    print(f"to {self.net.neuron[k].charge} by accumulating charge {self.events[i][k]} and leak charge {self.net.neuron[k].leak}")
                    #step 2: if charge < std_potential, charge=std_potential   
                    if (self.net.neuron[k].charge < self.net.neuron[k].std_potential):
                        print(f"Charge < std_potential. The neuron {k} charge has been reset from {self.net.neuron[k].charge}")
                        self.net.neuron[k].charge=self.net.neuron[k].std_potential; 
                        print(f"to {self.net.neuron[k].charge}")
                    #step 3: if charge > threshold, period="refr"    
                    if (self.net.neuron[k].charge>self.net.neuron[k].threshold):
                        print(f"charge > threshold. Period type refr")
                        self.net.neuron[k].period_type="abs_refr"    
                #For refr period
                elif (self.net.neuron[k].period_type=="abs_refr"):
                    if (self.net.neuron[k].abs_time<self.net.neuron[k].abs_period):
                        #if charge is not refr potential
                        print(f"Neuron {k} is in abs_refr in period {i}")
                        if (self.net.neuron[k].charge != self.net.neuron[k].refr_potential):
                            #last fire is this time
                            print(f"Neuron {k} fires at time {i}")
                            self.net.neuron[k].last_fire=i;
                            #total fire_count+1
                            self.net.neuron[k].fire_count +=1;      
                            self.spikes_[i,k]=1
                            "Potentiation"
                            print(f"Potentiation of all Syn {self.net.neuron[k].To}")
                            #For all the destination neuron
                            for l in range(len(self.net.neuron[k].To)):
    # #                             To neurons of all neurons
                                tup=self.net.neuron[k].To[l]
                                # add the spiking events
                                self.events[i][tup[1]]+= tup[2];
    #                             # delta = this neuron - lth to neuron
                                delta_to=self.net.neuron[k].last_fire-self.net.neuron[tup[1]].last_fire
                                if (self.net.neuron[tup[1]].stdp==True):
                                    #if delta_to > 0
                                    if ((delta_to>0) and (delta_to<5)):
                                        #weight is among processors weight limit
                                        if (self.proc.max_weight > (self.net.Syn[tup[0]].weight + self.proc.stdp_potentiation[self.net.Syn[tup[0]].pot_pointer])):
                                            #potentiation
                                            self.net.Syn[tup[0]].weight += self.proc.stdp_potentiation[self.net.Syn[tup[0]].pot_pointer];
                                            #update the tuple
                                            self.net.neuron[k].To[l]=(tup[0],tup[1],self.net.Syn[tup[0]].weight,tup[3]);
                                            #If all the pot_pointer not done
                                            if (self.net.Syn[tup[0]].pot_pointer<(len(self.proc.stdp_potentiation)-1)):
                                                self.net.Syn[tup[0]].pot_pointer += 1;
                            "Depression" 
                            print(f"Depression of all Syn {self.net.neuron[k].From}")
                            #For all the from neurons
                            for l in range(len(self.net.neuron[k].From)):
                                #tuple of neuron k of lth from neuron
                                tup = self.net.neuron[k].From[l];
                                #add the spiking events
                                self.events[i][tup[1]]-= tup[2];
                                #delta_from for all neurons
                                delta_from=self.net.neuron[k].last_fire-self.net.neuron[tup[1]].last_fire;
                                #if the neuron does not have STDP
                                if (self.net.neuron[tup[1]].stdp==True):
                                    if delta_from>0:
                                        if (self.proc.max_weight < (self.net.Syn[tup[0]].weight + self.proc.stdp_depression[self.net.Syn[tup[0]].dep_pointer])):
                                            self.net.Syn[tup[0]].weight += self.proc.stdp_depression[self.net.Syn[tup[0]].dep_pointer]
                                            self.net.neuron[k].To[l]=(tup[0],tup[1],self.net.Syn[tup[0]].weight,tup[3])
                                            if (self.net.Syn[tup[0]].dep_pointer<int(self.proc.stdp_depression.size()-1)):
                                                self.net.Syn[tup[0]].dep_pointer += 1 
                            self.net.neuron[k].charge=self.net.neuron[k].refr_potential
                        self.net.neuron[k].abs_time += 1
                    else:
                        self.net.neuron[k].charge = self.net.neuron[k].refr_potential
                        self.net.neuron[k].period_type="rel_refr"
                        self.net.neuron[k].abs_time = 0       
                        
                if self.net.neuron[k].period_type=="rel_refr":
                    print(f"The neuron {self.net.neuron[k].node_id} is in rel_refr in time {i}. So the charge has been changed from {self.net.neuron[k].charge}")
                    #Step 1: charge add and leak 
                    self.net.neuron[k].charge = self.net.neuron[k].charge + self.events[i][k]+self.net.neuron[k].refr_leak
                    print(f"to {self.net.neuron[k].charge} by accumulating charge {self.events[i][k]} and leak charge {self.net.neuron[k].refr_leak}")
                    #step 2: if charge < refr_potential, charge=refr_potential   
                    if (self.net.neuron[k].charge < self.net.neuron[k].refr_potential):
                        print(f"Charge < std_potential. The neuron {k} charge has been reset from {self.net.neuron[k].charge}")
                        self.net.neuron[k].charge=self.net.neuron[k].refr_potential; 
                        print(f"to {self.net.neuron[k].charge}")
                    #step 3: if charge > std_potential, period="std"    
                    if (self.net.neuron[k].charge>self.net.neuron[k].std_potential):
                        print(f"charge > threshold. Period type refr")
                        self.net.neuron[k].period_type="std"
                
                self.potential[i,k]=self.net.neuron[k].charge    