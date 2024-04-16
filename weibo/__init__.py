
def get_model(args): 
    
    if args.model_type == "jlogits":
        from weibo.joint_model import MultimodalWeiboModel
    elif args.model_type == "ensemble":
        from weibo.ensemble_model import MultimodalWeiboModel
    elif args.model_type == "qmf":
        from weibo.qmf_model import MultimodalWeiboModel
    elif args.model_type == "ogm_ge": 
        from weibo.ogm_ge_model import MultimodalWeiboModel
    else:   
        raise NotImplementedError("Model type not implemented")
    
    return MultimodalWeiboModel(args)

