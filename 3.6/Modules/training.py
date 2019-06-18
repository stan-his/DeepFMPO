import numpy as np
from global_parameters import MAX_SWAP, MAX_FRAGMENTS
from rewards import get_init_dist, evaluate_mol, modify_fragment
import logging

batch_size = 512

epochs = 4000

gamma = 0.95
times = 8
scores = 1. / times
n_actions = MAX_FRAGMENTS * MAX_SWAP + 1




def train(X, actor, critic, decodings,out_dir=None):
    
    hist = []
    dist = get_init_dist(X, decodings)
    m = X.shape[1]
    
    
    for e in range(epochs):

        rand_n = np.random.randint(0,X.shape[0],batch_size)
        batch_mol = X[rand_n].copy()
        r_tot = np.zeros(batch_size)
        org_mols = batch_mol.copy()
        stopped = np.zeros(batch_size) != 0

        for t in range(times):

            tm = (np.ones((batch_size,1)) * t) / times

            probs = actor.predict([batch_mol, tm])
            actions = np.zeros((batch_size))
            rand_select = np.random.rand(batch_size)
            old_batch = batch_mol.copy()
            rewards = np.zeros((batch_size,1))


            for i in range(batch_size):
                a = 0
                while True:
                    rand_select[i] -= probs[i,a]
                    if rand_select[i] < 0 or a + 1 == n_actions:
                        break
                    a += 1

                actions[i] = a 

            Vs = critic.predict([batch_mol,tm])

            for i in range(batch_size):

                a = int(actions[i])
                if stopped[i] or a == n_actions - 1:
                    stopped[i] = True
                    if t == 0:
                        rewards[i] += -1.

                    continue



                a = a // MAX_SWAP
                s = a % MAX_SWAP
                if batch_mol[i,a,0] == 1:
                    batch_mol[i,a] = modify_fragment(batch_mol[i,a], s)
                    # rewards[i] = scores * get_reward(batch_mol[i], e)
                else:
                    rewards[i] -= 0.1

            # If final round
            if t + 1 == times:
                frs = []
                for i in range(batch_mol.shape[0]):

                    # If new molecule
                    if not np.all(org_mols[i] == batch_mol[i]):

                        fr = evaluate_mol(batch_mol[i], e, decodings)
                        frs.append(fr)
                        rewards[i] += np.sum(fr * dist)

                        if all(fr):
                            rewards[i] *= 2
                    else:
    #                     rewards[i] += -1
                        frs.append([False] * 4)



                dist = 0.5 * dist + 0.5 * (0.25 * batch_size / (1.0 + np.sum(frs,0)))



            target = rewards + gamma * critic.predict([batch_mol, tm+1.0/times])
            td_error = target - Vs


            critic.fit([old_batch,tm], target, verbose=0)
            target_actor = np.zeros_like(probs)

            for i in range(batch_size):
                a = int(actions[i])
                loss = -np.log(probs[i,a]) * td_error[i]
                target_actor[i,a] = td_error[i]


            actor.fit([old_batch,tm], target_actor, verbose=0)

            r_tot += rewards[:,0]


        np.save("History/in-{}.npy".format(e), org_mols)
        np.save("History/out-{}.npy".format(e), batch_mol)
        np.save("History/score-{}.npy".format(e), np.asarray(frs))


        hist.append((np.mean(r_tot), np.mean(frs,0), np.mean(np.sum(frs, 1) == 4)))
        print ("Epoch {2} \t Mean score: {0:.3}\t\t Percentage in range: {1},  {3}".format(
            np.mean(r_tot), [round(x,2) for x in np.mean(frs,0)], e,
            round(np.mean(np.sum(frs, 1) == 4),2)
        ))
        
    return hist
