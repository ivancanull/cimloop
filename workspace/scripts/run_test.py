from html import parser
from ocim_utils import test_full_dnn, test_one_layer, test_hybrid_mapping
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--arch", type=str, default="proposed_mrr_1bit_input_delay_line")
    parser.add_argument("--network", type=str, default="alexnet")
    parser.add_argument("--run_test", type=str, default="full_dnn")
    parser.add_argument("--layer", type=str, default="0")

    args = parser.parse_args()      


    if args.run_test == "full_dnn":
        test_full_dnn(args.arch, args.network, "fetch_all_lpddr4")

    if args.run_test == "one_layer":
        test_one_layer(args.arch, args.network, args.layer, "fetch_all_lpddr4")

    if args.run_test == "hybrid_mapping":
        # test_hybrid_mapping(
        #     macro_regular="proposed_mrr_1bit_input_delay_line", 
        #     macro_hybrid="proposed_mrr_1bit_input_delay_line_wi", 
        #     mapping_config=f"/home/zhanghf/projects/cimloop/workspace/hybrid_mapping/alexnet.yaml", 
        #     dnn_name="alexnet", 
        #     system="fetch_all_lpddr4"
        # )
        test_hybrid_mapping(
            macro_regular="proposed_mrr_1bit_input_delay_line", 
            macro_hybrid="proposed_mrr_1bit_input_delay_line_wi", 
            mapping_config=f"/home/zhanghf/projects/cimloop/workspace/hybrid_mapping/resnet18.yaml", 
            dnn_name="resnet18", 
            system="fetch_all_lpddr4"
        )
    
    # test_architecture("proposed_mrr")
    # test_architecture("albireo_isca_2021_macro_only")
    # ocim_utils.test_one_layer("holylight_date_2019", "alexnet", "0", "fetch_all_lpddr4", "proposed_mrr")
    # ocim_utils.test_one_layer("proposed_mrr_1bit_input_delay_line", "alexnet", "0", "fetch_all_lpddr4", "proposed_mrr")
    
    ## test different glbs
    # ocim_utils.test_different_glbs("proposed", "alexnet", "fetch_all_lpddr4", "proposed_different_glbs")   
    
    # ocim_utils.test_two_layers("proposed", "alexnet", "fetch_all_lpddr4", "proposed")   

    # # alexnet
    # ocim_utils.test_full_dnn("proposed", "alexnet", "fetch_all_lpddr4", "proposed")
    
    # # mobilenet_v3
    # ocim_utils.test_full_dnn("proposed", "mobilenet_v3", "fetch_all_lpddr4", "proposed")
    # # resnet_18
    # ocim_utils.test_full_dnn("proposed", "resnet18_condensed", "fetch_all_lpddr4", "proposed")
    # # vgg_16
    # ocim_utils.test_full_dnn("proposed", "vgg16_condensed", "fetch_all_lpddr4", "proposed")

    # # gpt2_medium
    # ocim_utils.test_full_dnn("proposed", "gpt2_condensed", "fetch_all_lpddr4", "proposed")
