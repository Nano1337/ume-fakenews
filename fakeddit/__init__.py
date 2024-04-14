
def get_model(args): 
    
    if args.model_type == "jlogits":
        from fakeddit.joint_model import MultimodalFakedditModel
    elif args.model_type == "ensemble":
        from fakeddit.ensemble_model import MultimodalFakedditModel
    elif args.model_type == "qmf":
        from fakeddit.qmf_model import MultimodalFakedditModel
    elif args.model_type == "ogm_ge": 
        from fakeddit.ogm_ge_model import MultimodalFakedditModel
    else:   
        raise NotImplementedError("Model type not implemented")
    
    return MultimodalFakedditModel(args)