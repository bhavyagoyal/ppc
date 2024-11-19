# backup for peak finding function. Not needed now

# Find peaks and converts to point cloud
def peakpoints(nr, nc, K, bin_size, spad, gtvalid, Rtilt, rbins, intensity, peaks_post_processing=True, decompressed=False, gaussian_filter_pulse=False):
    xx = np.linspace(1, nc, nc)
    yy = np.linspace(1, nr, nr)
    x, y = np.meshgrid(xx, yy)
    cx, cy, fx, fy = camera_params(K)
    xa = (x - cx)/fx
    ya = (y - cy)/fy
    xa, ya = xa[:,:,np.newaxis], ya[:,:,np.newaxis]
    nt = spad.shape[2]
    # Removing first few bins
    spad[:,:,:5] = 0

    if(decompressed):
        # compress and decompress using truncated fourier
        spad_copy = copy.deepcopy(spad)
        spad = decompress(spad)
    elif(gaussian_filter_pulse):
        gf_pulse = np.zeros((5,5,22))
        gf_pulse[2,2,:] = pulse[0][0]
        gf_pulse = skimage.filters.gaussian(gf_pulse,sigma=1.0)
        gf_pulse = gf_pulse/gf_pulse.sum()
        spad = scipy.signal.convolve(spad, gf_pulse, mode='same')
    else:
        spad = scipy.signal.convolve(spad, pulse, mode='same')

    allpeaks = np.zeros((nr, nc, NUM_PEAKS_START))
    if(True):
        for ii in range(1, nr+1):
            for jj in range(1, nc+1):
                #peaks = scipy.signal.find_peaks(spad[ii-1, jj-1,:], distance=10, height=2)[0][:NUM_PEAKS_START]
                if(peaks_post_processing):
                    peaks = scipy.signal.find_peaks(spad[ii-1, jj-1,:], distance=10, height=0.3)[0][:NUM_PEAKS_START]
                else:
                    # Use height 0 as after convolve, spad can have very small negative numbers instead of 0
                    # it uses fourier transform for fast calculation
                    peaks = scipy.signal.find_peaks(spad[ii-1, jj-1,:], distance=10, height=0.)[0][:NUM_PEAKS_START]
                allpeaks[ii-1,jj-1,:len(peaks)]=peaks
    
        allpeaks = allpeaks.astype(int)
        density = spad[np.arange(nr)[:, np.newaxis, np.newaxis], np.arange(nc)[np.newaxis, :, np.newaxis], allpeaks]
    
        dp = np.stack([density, allpeaks])
        dpindex = dp[0,:,:,:].argsort(axis=-1)
        dpindex = dpindex[:,:,::-1]
     
        density = dp[0,:,:,:]
        density = density[np.arange(nr)[:, np.newaxis, np.newaxis], np.arange(nc)[np.newaxis, :, np.newaxis], dpindex]
        allpeaks = dp[1,:,:,:]
        allpeaks = allpeaks[np.arange(nr)[:, np.newaxis, np.newaxis], np.arange(nc)[np.newaxis, :, np.newaxis], dpindex]
    
    density = density[:,:,:NUM_PEAKS]
    allpeaks = allpeaks[:,:,:NUM_PEAKS].astype(int)
    #print('All Peaks ', np.count_nonzero(allpeaks==0))

    if(peaks_post_processing):
        maxdensity = density.max(axis=-1, keepdims=True)
        removepeaks = density<(maxdensity-0.5)
        density[removepeaks]=0.
        allpeaks[removepeaks]=0

    total_sampling_prob = density.sum(-1, keepdims=True)
    total_sampling_prob[total_sampling_prob<1e-9]=1
    sampling_prob = density/total_sampling_prob
    # we might drop a few points later, if they are farther than 65.356 depth
    # which would make sum non 1, but ignoring that for now.


    # Can remove points that are too close to camera
    # Few examples that I saw, 58 was the min bin count
    #removepeaks = allpeaks<50
    #density[removepeaks]=0.
    #allpeaks[removepeaks]=0



    #for ii in range(nr//2+10, nr//2+12):
    #    for jj in range(1, nc+1):
    #        rbin = rbins[ii-1,jj-1]-1
    #        peaks = allpeaks[ii-1, jj-1,:]
    #        plt.close()
    #        plt.figure().set_figwidth(24)

    #        plt.subplot(2,1,1)
    #        plt.bar(range(nt), spad[ii-1, jj-1,:], width=0.9)
    #        plt.scatter([rbin], [spad[ii-1, jj-1, rbin]], c='g', alpha=0.3)
    #        plt.scatter(peaks, [spad[ii-1, jj-1, peaks]], c='r', alpha=0.7)
    #        plt.text(100, 0, str(rbin) + " " + str(spad[ii-1, jj-1].max()) + " " + str(peaks) + " " + str(density[ii-1,jj-1]) )
    #        plt.subplot(2,1,2)
    #        plt.bar(range(nt), spad_copy[ii-1, jj-1,:], width=0.9)

    #        plt.savefig('plots_compress32_filteringpeaks_1_50/fig_' + str(ii-1) + '_' + str(jj-1)+ '_hist.png', dpi=500)

    #        plt.close()
    #        inten = intensity.copy()
    #        inten[ii-1, :]=1
    #        inten[:, jj-1]=1
    #        plt.imshow(inten)
    #        plt.savefig('plots_compress32_filteringpeaks_1_50/fig_' + str(ii-1) + '_' + str(jj-1)+ '_depth.png')


    # Only using this for visualization. These points are approximately correct depth
    correct = abs(np.repeat(rbins[:,:,np.newaxis], NUM_PEAKS, axis=-1) - allpeaks)<=CORRECTNESS_THRESH

    dists = tof2depth(allpeaks*bin_size)
    depths = dists/(xa**2 + ya**2 + 1)**0.5

    #print(allpeaks[73,313,:])
    #print(dists[73,313,:])
    #print(depths[73,313,:])
    #print(np.count_nonzero(depths==0))

    depths = depths*1000.
    depths = depths.astype(np.uint16)
    depths = (depths>>3)<<3
    depths = depths*gtvalid[:,:,np.newaxis]

    depths = (depths>>3 | np.uint16(depths<<13))
    depths = depths.astype('float32')/1000.
    depths[depths>8]=8
    X = xa*depths
    Y = ya*depths
    Z = depths

    #AA = set(zip(*np.nonzero(Z==0)))
    #AA2 = set(zip(*np.nonzero(X==0)))
    #AA3 = set(zip(*np.nonzero(Y==0)))
    #BB = set(zip(*np.nonzero(dists==0)))
    #gtvalidthree = np.tile(gtvalid[:,:,None],3)
    #CC = set(zip(*np.nonzero(gtvalidthree==0)))
    #DD = AA - (CC|BB)
    #DD2 = AA2 - (CC|BB)
    #DD3 = AA3 - (CC|BB)
    #print(len(BB), sorted(list(BB)))
    #print(len(AA), len(BB), len(CC))
    #print(len(DD), sorted(list(DD)))
    #print(len(DD2), sorted(list(DD2)))
    #print(len(DD3), sorted(list(DD3)))
    #print(depths[73,313,:])

    points3d = np.stack([X, Z, -Y])
    points3d, density, sampling_prob = points3d.reshape((3,-1)), density.flatten(), sampling_prob.flatten()
    points3d = np.matmul(Rtilt, points3d)
    return points3d, density, sampling_prob, correct, np.tile(xa, NUM_PEAKS).flatten(), np.tile(ya, NUM_PEAKS).flatten()





# Usage
points3d, density, sampling_prob, correct, xa, ya = peakpoints(nr, nc, K, data['bin_size'], spad, gtvalid, Rtilt, data['range_bins'], data['intensity'])





            if(sampling_prob is not None):
                sampling_prob = sampling_prob[valid]
                xa,ya = xa[valid], ya[valid]

                # xa, ya is pixel cordinates
                # Sampling num_points pixels and then selecting all peak from those pixels.
                # so, this allows sampling number of pixels rather than points.
                xya = list( zip(xa,ya) )
                all_xya = list( set( xya ) )
                if(len(all_xya)<=num_points):
                    selected_xy = set(all_xya)
                else:
                    selected_xy = set(random.sample(all_xya, num_points))
                choices = []
                for xy_idx, xy in enumerate(xya):
                    if(xy in selected_xy):
                        choices.append(xy_idx)
                assert len(choices)>=num_points
                points3d = points3d[choices]

                # Earlier I was sampling by sampling_prob, but this does not ensure all peaks from a pixel are included
                #points3d, choices = random_sampling(points3d, num_points, p=sampling_prob/sampling_prob.sum())
                #negprobs = -1*sampling_prob[choices]
                #newchoices = negprobs.argsort()
                #choices = choices[newchoices]
                #points3d = points3d[newchoices]
                sampling_prob = sampling_prob[choices]
            else:



#sys.path.append('../../../spatio-temporal-csph/')
#from csph_layers import CSPH3DLayer


NUM_PEAKS=3 # upto NUM_PEAKS peaks are selected
NUM_PEAKS_START = 110
 
nt_compression = 1024
csph1D_obj = None

# Does compression and decompression
def decompress(spad):
    assert spad.shape[2]==nt_compression
    spad_out = spad.transpose(2,0,1)
    spad_out = torch.from_numpy(spad_out[None,None,...])
    spad_out = csph1D_obj(spad_out)[0,0,...].numpy()
    spad_out = spad_out.transpose(1,2,0)
    return spad_out


def argmaxfiltering(spad):
    spaddensity = scipy.signal.convolve(spad, pulse, mode='same')
    return spaddensity.argmax(-1), spaddensity.max(-1)


def argmaxdecompressed(spad):
    spad = decompress(spad)
    return spad.argmax(-1)


# random tie breaker for argmax on raw histogram bins
def argmaxrandomtie(spad):
    maxval = spad.max(axis=-1, keepdims=True)
    maxmatrix = spad == maxval
    
    spadmax = np.zeros(spad.shape[:2], dtype=np.int32)
    for i in range(spad.shape[0]):
        for j in range(spad.shape[1]):
            spadmax[i,j] = np.random.choice(np.flatnonzero(maxmatrix[i,j,:]))
    return spadmax





    if('decompress' in args.method):
        csph1D_obj = CSPH3DLayer(k=32, num_bins=nt_compression, tblock_init='TruncFourier', optimize_codes=False, encoding_type='csph1d', zero_mean_tdim_codes=True)
        csph1D_obj.to(device='cpu')




            #for ii in range(nr//2+10, nr//2+15):
            #    for jj in range(1, nc+1):
            #        plt.close()
            #        plt.figure().set_figwidth(14)
            #        #plt.bar(range(nt), spadcopy[ii-1, jj-1,:], width=0.9)
            #        plt.bar(range(300), spadcopy[ii-1, jj-1,:300], width=0.9)
            #        rbin = data['range_bins'][ii-1,jj-1]-1
            #        selected = spad[ii-1, jj-1]
            #        #plt.scatter([rbin], [spadcopy[ii-1, jj-1, rbin]], c='g', alpha=0.3)
            #        #plt.scatter([selected], [spadcopy[ii-1, jj-1, selected]], c='r', alpha=0.7)
            #        #plt.text(100, 0.4, str(rbin) + " " + str(spadcopy[ii-1, jj-1, :].max()) + " " + str(selected) + str(spadcopy[ii-1, jj-1, selected]) + " " + str(gtvalid[ii-1,jj-1]) + " " + str(data['intensity'][ii-1,jj-1]))
            #        plt.xlabel('Time (ns)')
            #        #plt.xticks([x*30 for x in range(10)], [x*30*data['bin_size']*1000000000 for x in range(10)])
            #        plt.xticks([x*60 for x in range(5)], [x*40 for x in range(5)])
            #        ax = plt.gca()
            #        ax.set_ylim([0,8])
            #        plt.ylabel('Photon Count')
            #        plt.tight_layout()
            #        plt.savefig('plots_argmax_filtering_sbr_' + args.sbr + '/fig' + str(ii-1) + '_' + str(jj-1)+ '_hist.png', dpi=500)
            #        plt.close()

            #        #inten = data['intensity'].copy()
            #        #inten[ii-1, :]=1
            #        #inten[:, jj-1]=1
            #        #plt.imshow(inten)
            #        #plt.savefig('plots_argmax_filtering_sbr_' + args.sbr + '/fig' + str(ii-1) + '_' + str(jj-1)+ '_depth.png')

