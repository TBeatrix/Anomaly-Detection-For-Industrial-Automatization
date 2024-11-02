manual_prompts = {
    'stacked': [
        ['Crumpled paper package on top. ', 'stacked boxes'],
        ['package material on top.', 'stacked boxes'],
        ['paper sheet on top.', 'stacked boxes'],
        ['red knife.', 'box cutter.', 'stacked boxes'],
        ['anything other than boxes next to each other.', 'stacked boxes'],
         ['hands.', 'stacked boxes'],
        ['fingers. ', 'stacked boxes'],
        ['arm.', 'stacked boxes'],
        ['opened box. ', 'stacked boxes']
    ],

    'side_reels_small': [   
        ['paper package on top. plastic package on top.', ' stacked reels'],
        ['red knife', 'red box cutter.', 'stacked reels'],
        ['black reels.', 'stacked reels'],
        ['circular object. a reel disc lays on top.', 'stacked reels'],
        ['hands. ', 'stacked reels'],
        ['fingers.', 'stacked reels'],
        ['glove. ', 'stacked reels'],
        ['skin. arm.', 'stacked reels'],   
    ],

    'side_reels_big': [
    
        ['paper package on top.', ' stacked reels'],
        ['black reels.', 'stacked reels'],
        ['circular object. ', 'stacked reels'],
        ['a reel disc lays on top.', 'stacked reels'],
        [' plastic package on top.', 'stacked reels'],     
    ],

    'reels_small': [
        ['package material on top. ' , 'reel disc'],
        ['paper package on top.', 'reel disc'],      
        ['narrow rubber band around it.', 'reel disc'],
        ['hands.', 'reel disc'],
        ['fingers. ', 'reel disc'],
        ['gloves. ', 'reel disc'],
        ['skin. arm.', 'reel disc'],
    ],

    
    'reels_big': [
        ['paper package on top.', 'reel disc'],      
        ['red package.', 'reel disc'],
        ['not circular.', 'reel disc'],
        ['vacuumed packed. ', 'reel disc'],
        ['in a metalized foil  package. ', 'reel disc'],
    ],

    'pizza_boxes': [
        ['plastic package on top.', 'pizza box shaped box'],
        ['paper sheet on top.', 'pizza box shaped box'],
        ['red knife. ', 'pizza box shaped box'],
        ['hands.', 'pizza box shaped box'],
        ['fingers. ', 'pizza box shaped box'],
        ['gloves. ', 'pizza box shaped box'],
        ['skin. arm.', 'pizza box shaped box'],
        ['paper package next to. ', 'pizza box shaped box']
    ],

    'PCBs_small': [
        ['fingers.', 'metalized foil bag'],
        ['hands.', 'metalized foil bag'],
        ['gloves ', 'metalized foil bag'],
        [' skin. arm.', 'metalized foil bag'],
       
    ],

    'PCBs_big': [
        ['polystyrene  material on top', 'item in a metalized foil bag'],
        ['paper package on top. ', 'item in a metalized foil bag'],
        ['plastic package on top .', 'item in a metalized foil bag'],
        ['empty box. the foil bag is missing', 'item in a metalized foil bag']
    ],

}

property_prompts = {
    'stacked': 'the image of boxes have 1 dissimilar stacked_boxes, with a maximum of 3 anomaly. The anomaly would not exceed 0.6 object area. ',
    'side_reels_small': 'the image of reels have 1 dissimilar stacked_reels, with a maximum of 2 anomaly. The anomaly would not exceed 0.35 object area. ',
    'side_reels_big': 'the image of reels have 1 dissimilar stacked_reels, with a maximum of 2 anomaly. The anomaly would not exceed 0.8 object area. ',
    'reels_big':  'the image of reels have 1 dissimilar reel, with a maximum of 2 anomaly. The anomaly would not exceed 1.0 object area. ',
    'pizza_boxes':'the image of box have 1 dissimilar box, with a maximum of 2 anomaly. The anomaly would not exceed 0.5 object area. ',
    'PCBs_small': 'the image of PCB have 1 dissimilar metalized_foil_bag, with a maximum of 3 anomaly. The anomaly would not exceed 0.4 object area. ',
    'PCBs_big': 'the image of PCB have 1 dissimilar item_in_a_metalized_foil_bag, with a maximum of 2 anomaly. The anomaly would not exceed 1.0 object area. ',
    'empty_box': 'the image of box have 1 dissimilar box, with a maximum of 3 anomaly. The anomaly would not exceed 0.4 object area. ',
    
}
