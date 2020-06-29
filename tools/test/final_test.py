
from test_utils import * 




def save_predictions(start = 200, end = 210, path = 'images/full_loop/binaries_scores'):
    infos = get_info(start = start, end = end ) 
    image_paths = np.array([infos[i][0] for i in range(len(infos))])

    binaries, scores = predict(image_paths, path)  

    return image_paths, binaries, scores




def test() :
    #save_predictions() 
    
    predictions_list = get_image_paths_list(predictions_path) 
    predictions_list = [i for i in predictions_list if 'postprocess' not in i] 
    names = [os.path.split(prediction)[1].split('.')[0] for prediction in predictions_list] 
    

    pp = PostProcessor() 

    for i, prediction in enumerate(predictions_list) : 
        binary = read_image(prediction) 
        ret = pp.process(binary) 
        save_image('images2/scores4', 'postprocess_{}'.format(names[i]), ret['perfect_mask'])




def mask_on_source() :
    predictions_list = get_image_paths_list(predictions_path) 
    predictions_list = [i for i in predictions_list if 'postprocess_score' in i]  
    names = [os.path.split(prediction)[1].split('.')[0] for prediction in predictions_list] 
    
    infos = get_info(start = 400, end = 410) 

    for i in range(len(infos)):
        image = read_image(infos[i][0]) 
        mask = read_image(predictions_list[i]) 

        image |= mask 

        save_image('images/scores4', 'source_pure_score_{}'.format(i), image) 



def full_loop(path = 'images/full_loop') : 
    
    sources, binaries, scores = save_predictions(path = os.path.join(path,'binaries_scores')) 

    pp = PostProcessor()
    processing_time = [] 

    for i, source in enumerate(sources) :
        source = read_image(source) 
        save_image(path+'/sources', 'source_{}'.format(i), source) 

        start = time.time() 
        score_ret = pp.process(read_image(scores[i])) 
        binary_ret = pp.process(read_image(binaries[i])) 
        processing_time.append(time.time() - start) 

        save_image(path+'/masks', 'binary_{}'.format(i), binary_ret['mask']) 
        save_image(path+'/masks', 'score_{}'.format(i), score_ret['mask']) 
        
        save_image(path+'/masks', 'binary_prefect_{}'.format(i), binary_ret['perfect_mask']) 
        save_image(path+'/masks', 'score_prefect_{}'.format(i), score_ret['perfect_mask']) 
        
        score_source = source.copy() 
        score_perfect_source = source.copy() 
        binary_source = source.copy() 
        binary_perfect_source = source.copy() 

        score_source |= score_ret['mask'] 
        score_perfect_source |= score_ret['perfect_mask']
        binary_source |= binary_ret['mask'] 
        binary_perfect_source |= binary_ret['perfect_mask'] 

        save_image(path+'/mask_on_sourcce', 'score_{}'.format(i), score_source) 
        save_image(path+'/mask_on_sourcce', 'score_perfect_{}'.format(i), score_perfect_source) 
        save_image(path+'/mask_on_sourcce', 'binary_{}'.format(i), binary_source) 
        save_image(path+'/mask_on_sourcce', 'binary_perfect_{}'.format(i), binary_perfect_source) 



    print("processed {}    in {}    average {}    max {}".format(len(sources), np.sum(processing_time), np.mean(processing_time)/2, np.max(processing_time)/2))


full_loop() 