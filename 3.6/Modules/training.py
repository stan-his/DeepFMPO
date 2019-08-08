import numpy as np
from global_parameters import MAX_SWAP, MAX_FRAGMENTS, GAMMA, BATCH_SIZE, EPOCHS, TIMES, FEATURES
from rewards import get_init_dist, evaluate_mol, modify_fragment
import logging


scores = 1. / TIMES
n_actions = MAX_FRAGMENTS * MAX_SWAP + 1



# Train actor and critic networks
def train(X, actor, critic, decodings, out_dir=None):

    hist = []
    dist = get_init_dist(X, decodings)
    m = X.shape[1]


    # For every epoch
    for e in range(EPOCHS):

        # Select random starting "lead" molecules
        rand_n = np.random.randint(0,X.shape[0],BATCH_SIZE)
        batch_mol = X[rand_n].copy()
        r_tot = np.zeros(BATCH_SIZE)
        org_mols = batch_mol.copy()
        stopped = np.zeros(BATCH_SIZE) != 0


        # For all modification steps
        for t in range(TIMES):


            tm = (np.ones((BATCH_SIZE,1)) * t) / TIMES

            probs = actor.predict([batch_mol, tm])
            actions = np.zeros((BATCH_SIZE))
            rand_select = np.random.rand(BATCH_SIZE)
            old_batch = batch_mol.copy()
            rewards = np.zeros((BATCH_SIZE,1))


            # Find probabilities for modification actions
            for i in range(BATCH_SIZE):

                a = 0
                while True:
                    rand_select[i] -= probs[i,a]
                    if rand_select[i] < 0 or a + 1 == n_actions:
                        break
                    a += 1

                actions[i] = a

            # Initial critic value
            Vs = critic.predict([batch_mol,tm])


            # Select actions
            for i in range(BATCH_SIZE):

                a = int(actions[i])
                if stopped[i] or a == n_actions - 1:
                    stopped[i] = True
                    if t == 0:
                        rewards[i] += -1.

                    continue



                a = int(a // MAX_SWAP)

                s = a % MAX_SWAP
                if batch_mol[i,a,0] == 1:
                    batch_mol[i,a] = modify_fragment(batch_mol[i,a], s)
                else:
                    rewards[i] -= 0.1

            # If final round
            if t + 1 == TIMES:
                frs = []
                for i in range(batch_mol.shape[0]):

                    # If molecule was modified
                    if not np.all(org_mols[i] == batch_mol[i]):

                        fr = evaluate_mol(batch_mol[i], e, decodings)
                        frs.append(fr)
                        rewards[i] += np.sum(fr * dist)

                        if all(fr):
                            rewards[i] *= 2
                    else:
                        frs.append([False] * FEATURES)


                # Update distribution of rewards
                dist = 0.5 * dist + 0.5 * (1.0/FEATURES * BATCH_SIZE / (1.0 + np.sum(frs,0)))


            # Calculate TD-error
            target = rewards + GAMMA * critic.predict([batch_mol, tm+1.0/TIMES])
            td_error = target - Vs

            # Minimize TD-error
            critic.fit([old_batch,tm], target, verbose=0)
            target_actor = np.zeros_like(probs)


            for i in range(BATCH_SIZE):

                a = int(actions[i])
                loss = -np.log(probs[i,a]) * td_error[i]
                target_actor[i,a] = td_error[i]

            # Maximize expected reward.
            actor.fit([old_batch,tm], target_actor, verbose=0)

            r_tot += rewards[:,0]


        np.save("History/in-{}.npy".format(e), org_mols)
        np.save("History/out-{}.npy".format(e), batch_mol)
        np.save("History/score-{}.npy".format(e), np.asarray(frs))


        hist.append([np.mean(r_tot)] + list(np.mean(frs,0)) + [np.mean(np.sum(frs, 1) == 4)])
        print ("Epoch {2} \t Mean score: {0:.3}\t\t Percentage in range: {1},  {3}".format(
            np.mean(r_tot), [round(x,2) for x in np.mean(frs,0)], e,
            round(np.mean(np.sum(frs, 1) == 4),2)
        ))
        

    return hist
