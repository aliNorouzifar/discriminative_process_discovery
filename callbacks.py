from dash import Input, Output, State, html
from pages.cluster_discovery import IM_check_all,IMr4clusters, IM_parameters_show, run_clustering, run_discriminator_visualize_rules,clustering_params_block, rule_src_selection,show_rule_uploder,show_rule_uploder2,show_Minerful_params, IMr_params_show, rule_related_statistics_show,conformance_related_statistics_show,show_petri_net,IMr_no_rules_params_show
from prolysis.analysis.EMD_based_framework import apply_EMD,apply_segmentation, export_logs
from prolysis.analysis.explainability_extraction import decl2NL, apply_X, apply_feature_extraction, generate_features
# from pages.XPVI import parameters_view_PVI, PVI_figures_EMD,PVI_figures_Segments,parameters_feature_extraction, XPVI_figures, decl2NL_parameters, statistics_print, parameters_view_segmentation
import os
import shutil
from prolysis.calls.minerful_calls import discover_declare
from pathlib import Path
from prolysis.discovery.discovery import run_IMr_multi_rule
import json
from prolysis.analysis.evaluation import conformance_checking,extract_significant_dev, conformance_checking_bi
from prolysis.util.redis_connection import redis_client
from prolysis.rules_handling.utils import rules_from_json
from prolysis.util.logging import log_command
from prolysis.util.utils import import_log
import numpy as np




UPLOAD_FOLDER = "event_logs"
OUTPUT_FOLDER = "output_files"
WINDOWS = []


def clear_upload_folder(folder_path):
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def register_callbacks(app):
    clear_upload_folder("event_logs")
    clear_upload_folder("output_files")

    @app.callback(
        Output("plot_block_clustering", "children"),
        [Input("upload-Lp", "isCompleted")],
        [Input("upload-Lm", "isCompleted")],
        State("upload-Lp", "upload_id"),
        State("upload-Lm", "upload_id"),
    )
    def parameters_PVI(isCompletedP,isCompletedM,input_fileP,input_fileM):
        if isCompletedP==True and isCompletedM:
            input_log_pathP = os.path.join(UPLOAD_FOLDER, input_fileP)
            filesP = os.listdir(input_log_pathP) if os.path.exists(input_log_pathP) else []
            fileP = Path(UPLOAD_FOLDER) / f"{input_fileP}" / filesP[0]
            input_log_pathM = os.path.join(UPLOAD_FOLDER, input_fileM)
            filesM = os.listdir(input_log_pathM) if os.path.exists(input_log_pathM) else []
            fileM = Path(UPLOAD_FOLDER) / f"{input_fileM}" / filesM[0]
            return run_discriminator_visualize_rules(fileP,fileM,OUTPUT_FOLDER)

    @app.callback(
        Output('output-data-upload2', 'children'),
        Input('clustering_params', "n_clicks"),
    )
    def clustering_params_show(n):
        if n>0:
            return clustering_params_block()

    @app.callback(
        Output('statistics_clustering', 'children'),
        Input('clustering_run', "n_clicks"),
        State('no_clusters','value')
    )
    def clustering_params_show(n,n_clusters):
        if n>0:
            Z = np.array(json.loads(redis_client.get('Z_clustering')))
            html_stats, json_report = run_clustering(n_clusters, Z, OUTPUT_FOLDER)
            with open(os.path.join(OUTPUT_FOLDER,"output.json"), "w") as f:
                json.dump(json_report, f, indent=4)
            return html_stats

    @app.callback(
        Output('output-data-upload4', 'children'),
        Input('discover_models', "n_clicks"),
        State('no_clusters', 'value'),
    )
    def IM_parameters(n,n_clusters):
        if n>0:
            return IM_parameters_show(n_clusters)

    @app.callback(
        Output('overal_insights', 'children'),
        Input('discover_all', "n_clicks"),
        State('no_clusters', 'value'),
        State("upload-Lp", "upload_id"),
        State("upload-Lm", "upload_id"),
    )
    def IM_parameters(n,n_clusters, log_file_p, log_file_m):
        if n>0:
            return IM_check_all(n_clusters,OUTPUT_FOLDER,UPLOAD_FOLDER, log_file_p, log_file_m)

    @app.callback(
        Output('models_clusters', 'children'),
        Input('run_IMr_clusters', "n_clicks"),
        State('cluster_selected', 'value'),
        State('support_val', 'value'),
        State('confidence_val', 'value'),
        State('sup_IMr_val', 'value'),
    )
    def IM_parameters(n,cluster_selected,support_Minerful, confidence_Minerful, support):
        if n > 0:
            return IMr4clusters(cluster_selected,support_Minerful, confidence_Minerful, support, OUTPUT_FOLDER)

    @app.callback(
        Output('stats_cluster_box', 'children'),
        Input('stats_show_cl', "n_clicks"),
        State('cluster_selected', 'value'),
        State("upload-Lp", "upload_id"),
        State("upload-Lm", "upload_id"),
    )
    def rule_related_statistics_cl(n4,cluster_selected, log_file_p, log_file_m):
        if n4 > 0:
            input_log_path_Lp = os.path.join(UPLOAD_FOLDER, log_file_p)
            files_Lp = os.listdir(input_log_path_Lp) if os.path.exists(input_log_path_Lp) else []
            log_path_Lp = Path(UPLOAD_FOLDER) / f"{log_file_p}" / files_Lp[0]

            input_log_path_Lm = os.path.join(UPLOAD_FOLDER, log_file_m)
            files_Lm = os.listdir(input_log_path_Lm) if os.path.exists(input_log_path_Lm) else []
            log_path_Lm = Path(UPLOAD_FOLDER) / f"{log_file_m}" / files_Lm[0]

            model_path = os.path.join(OUTPUT_FOLDER, f"model_{cluster_selected}.pnml")
            rep_dict = conformance_checking_bi(log_path_Lp, log_path_Lm, model_path)
            return conformance_related_statistics_show(rep_dict)
        return ""

    # def IM_parameters(n, cluster_selected):
    #     if n > 0:
    #         return IMr4clusters(cluster_selected, support_Minerful, confidence_Minerful, support, OUTPUT_FOLDER)


    # Callback to update the output based on the selected options
    @app.callback(
        Output('output-data-upload6', 'children'),
        Output('show_IMr_run1', 'data'),
        Output('des_upload', 'children'),
        Input('rule_src', 'value')
    )
    def update_output(rule_source):
        if not os.path.exists(os.path.join(r"event_logs", "rules")):
            os.makedirs(os.path.join(r"event_logs", "rules"))
        else:
            clear_upload_folder(os.path.join(r"event_logs", "rules"))
        if rule_source=="manual":
            return show_rule_uploder(), False, show_rule_uploder2()
        elif rule_source=="Minerful":
            return show_Minerful_params(), False, show_rule_uploder2()
        elif rule_source=="no_rule":
            redis_client.set('rules', json.dumps([]))
            redis_client.set('dimensions', json.dumps([]))
            redis_client.set('activities', json.dumps([]))
            # return IMr_no_rules_params_show(), True
            return "", True, show_rule_uploder2()
        return "Select a rule source!", False, show_rule_uploder2()

    @app.callback(
        # Output('output-data-upload8', 'children'),
        Output('show_IMr_run2', 'data'),
        Input('run_Minerful', "n_clicks"),
        State("upload-Lp", "upload_id"),
        State("support_val", "value"),
        State("confidence_val", "value"),
    )
    def Minerful_call(n, input_file, support, confidence):
        if n>0:
            input_log_path = os.path.join(UPLOAD_FOLDER, input_file)
            files = os.listdir(input_log_path) if os.path.exists(input_log_path) else []
            file = Path(UPLOAD_FOLDER) / f"{input_file}" / files[0]
            output_log_path = os.path.join("event_logs", "rules", "rules.json")
            discover_declare(file, output_log_path, support, confidence)
            rules, activities = rules_from_json(str(output_log_path))
            redis_client.set('rules',json.dumps(rules))
            redis_client.set('dimensions',json.dumps(list(rules[0].keys()-['template', 'parameters'])))
            redis_client.set('activities', json.dumps(list(activities)))
            # return IMr_params_show(), True
            return True


###################
    @app.callback(
        # Output("output-data-upload10", "children"),
        Output('show_IMr_run3', 'data'),
        [Input("rule_upload", "isCompleted")],
        [State("upload-Lp", "upload_id")],
    )
    def parameters_PVI(isCompleted, id):
        if isCompleted == True:
            input_rule_path = os.path.join(UPLOAD_FOLDER, "rules")
            files = os.listdir(input_rule_path) if os.path.exists(input_rule_path) else []
            # rule_path = Path(UPLOAD_FOLDER) / f"{rule_file}" / files[0]
            rule_path = os.path.join(UPLOAD_FOLDER, "rules", files[0])
            # output_log_path = os.path.join("event_logs", "rules", "rules.json")
            rules, activities = rules_from_json(str(rule_path))
            redis_client.set('rules', json.dumps(rules))
            redis_client.set('dimensions', json.dumps(list(rules[0].keys() - ['template', 'parameters'])))
            redis_client.set('activities', json.dumps(list(activities)))
            # return IMr_params_show(),True
            return True


    @app.callback(
        # Output("output-data-upload10", "children"),
        # Output('show_IMr_run3', 'data'),
        Output('dummy-div', 'children'),
        [Input("rule_upload_des", "isCompleted")],
    )
    def des_rule_to_Redis(isCompleted):
        if isCompleted == True:
            input_rule_path = os.path.join(UPLOAD_FOLDER, "rules_des")
            files = os.listdir(input_rule_path) if os.path.exists(input_rule_path) else []
            # rule_path = Path(UPLOAD_FOLDER) / f"{rule_file}" / files[0]
            rule_path = os.path.join(UPLOAD_FOLDER, "rules_des", files[0])
            # output_log_path = os.path.join("event_logs", "rules", "rules.json")
            rules, activities = rules_from_json(str(rule_path))
            redis_client.set('rules_des', json.dumps(rules))
            redis_client.set('dimension_des', "desirability")
            # redis_client.set('activities', json.dumps(list(activities)))
            # return IMr_params_show(),True
            return True

    @app.callback(
        Output('output-data-upload10', 'children'),
        [Input('show_IMr_run1', 'data'),
        Input('show_IMr_run2', 'data'),
        Input('show_IMr_run3', 'data')])
    def sss(f1,f2,f3):
        if f1 ==True:
            return IMr_no_rules_params_show()
        elif f2==True:
            return IMr_params_show()
        elif f3==True:
            return IMr_params_show()


    @app.callback(
        Output("petri_net1", "children"),
        Input('run_IMr_selector', "n_clicks"),
        State("upload-Lp", "upload_id"),
        State("sup_IMr_val", "value"),
        State("dimension", "value"),
        State("absence_thr", "value"),
    )
    def IMr_call_rules(n1, log_file, sup, dim, abs_thr):
        if dim==None:
            dim = ""
        if abs_thr==None:
            abs_thr=""

        abs_thr = 0.01
        input_log_path = os.path.join(UPLOAD_FOLDER, log_file)
        files = os.listdir(input_log_path) if os.path.exists(input_log_path) else []
        log_path = Path(UPLOAD_FOLDER) / f"{log_file}" / files[0]
        rules = json.loads(redis_client.get('rules'))
        rules_des = json.loads(redis_client.get('rules_des'))
        activities = json.loads(redis_client.get('activities'))
        if n1>0:
            gviz = run_IMr_multi_rule(log_path, sup, rules,rules_des, activities, dim, "desirability", abs_thr,25)
            # (LPlus_LogFile, support, rules_cont, rules_des, activities, dim_cont, dim_des, abs_thr)
            return show_petri_net(gviz)
        return ""

    @app.callback(
        Output("gv", "style"),  # Update the style property of the graph
        Input("zoom-slider", "value"),  # Listen to the slider value
    )
    def update_zoom(zoom_value):
        # Dynamically update the CSS `transform: scale()` property
        return {"transform": f"scale({zoom_value})", "transformOrigin": "0 0"}

    # @app.callback(
    #     Output('output-data-upload5', 'children'),
    #     Input('stats_show', "n_clicks"),
    # )
    # def rule_related_statistics(n3):
    #     if n3>0:
    #         # Open and read the JSON file
    #         with open(os.path.join(r"output_files/", "stats.json"), "r") as file:
    #             data = json.load(file)
    #         dev_rank = extract_significant_dev(data["dev_list"])
    #         return rule_related_statistics_show(data["N.rules"], data["N.dev"], data["support_cost"], data["confidence_cost"],dev_rank)
    #     return ""

    @app.callback(
        Output('output-data-upload7', 'children'),
        Input('stats_show', "n_clicks"),
        State("upload-Lp", "upload_id"),
        State("upload-Lm", "upload_id"),
    )
    def rule_related_statistics(n4,log_file_p,log_file_m):
        if n4>0:
            input_log_path_Lp = os.path.join(UPLOAD_FOLDER, log_file_p)
            files_Lp = os.listdir(input_log_path_Lp) if os.path.exists(input_log_path_Lp) else []
            log_path_Lp = Path(UPLOAD_FOLDER) / f"{log_file_p}" / files_Lp[0]

            input_log_path_Lm = os.path.join(UPLOAD_FOLDER, log_file_m)
            files_Lm = os.listdir(input_log_path_Lm) if os.path.exists(input_log_path_Lm) else []
            log_path_Lm = Path(UPLOAD_FOLDER) / f"{log_file_m}" / files_Lm[0]


            model_path = os.path.join(r"output_files", "model.pnml")
            rep_dict = conformance_checking_bi(log_path_Lp,log_path_Lm, model_path)
            return conformance_related_statistics_show(rep_dict)
        return ""

    @app.callback(
        Input('remove_inputs', "n_clicks"),
        prevent_initial_call=True
    )
    def remove_inputs(n):
        if n>0:
            if not os.path.exists(r"event_logs"):
                os.makedirs(r"event_logs")
            else:
                clear_upload_folder(r"event_logs")





################## XPVI ########################
    @app.callback(
        Output("output-data-upload102", "children"),
        [Input("event_log_upload", "isCompleted")],
        [State("event_log_upload", "upload_id")],
    )
    def parameters_PVI(isCompleted, id):
        if isCompleted == True:
            folder_path = os.path.join(UPLOAD_FOLDER, id)
            files = os.listdir(folder_path) if os.path.exists(folder_path) else []
            file = Path(UPLOAD_FOLDER) / f"{id}" / files[0]
            print(file)
            log_command(f"Event Log {file} is going to be imported!")
            max_par, columns = import_log(file)
            log_command(f"Event Log {file} is imported!")
            return parameters_view_PVI(max_par, columns)

    '''significant distance parameter'''

    @app.callback(
        Output("output-data-upload104", "children"),
        [Input("Seg_parameters", "n_clicks")],
    )
    def parameters_segmentation(n):
        # print(redis_client.get('ali'))
        if n > 0:
            max_dist = float(redis_client.get("max_dist"))
            return parameters_view_segmentation(max_dist)

    '''applying the EMD-based process variant identification'''

    @app.callback(
        Output("output-data-upload103", "children"),
        Input("n_bins", "value"),
        Input("w", "value"),
        Input("kpi", "value")
    )
    def plot_data_EMD(n_bin, w, kpi):
        if kpi is not None:
            fig1 = apply_EMD(n_bin, w, kpi)
            return PVI_figures_EMD(fig1)

    '''pairwise comparison of the segments visualization'''

    @app.callback(
        Output("output-data-upload105", "children"),
        # Input("run_seg", "n_clicks"),
        State("n_bins", "value"),
        State("w", "value"),
        Input("sig_dist", "value"),
    )
    def plot_data_Segments(n_bin, w, sig):
        # if n>0:
        # fig_src1,fig_src2 = PVI_apply(n_bin, w, sig, faster, export, kpi, WINDOWS)
        fig2, peak_explanations = apply_segmentation(n_bin, w, sig)
        return PVI_figures_Segments(fig2, peak_explanations)

    '''segments_ export'''

    @app.callback(Output("output-data-upload111", "children"),
                  Input("export", "n_clicks")
                  )
    def export_logs_func(n):
        if n > 0:
            segments_ids = json.loads(redis_client.get("segments_ids"))
            log_command("exporting event logs started!")
            export_logs(segments_ids)
            log_command("exporting event logs done!")
            return "Event logs are exported!"

    '''Feature space generation (calling Minerful)'''

    @app.callback(
        Output("output-data-upload106", "children"),
        Input("X_parameters", "n_clicks"),
        State("w", "value"),
        State("kpi", "value"),
        State("n_bins", "value"),
    )
    def parameters_explainability(n, w, kpi, n_bin):
        if n > 0:
            log_command("event log is sent to Minerful for feature generation!")
            generate_features(w, kpi, n_bin)
            log_command("feature generation done!")
            return parameters_feature_extraction()

    '''Explainability results visualizations'''

    @app.callback(
        Output("output-data-upload107", "children"),
        # Input("minerful_run", "n_clicks"),
        State("n_bins", "value"),
        State("w", "value"),
        Input("theta_cvg", "value"),
        Input("n_clusters", "value")
    )
    def parameters_explainability(n_bin, w, theta_cvg, n_clusters):
        # if n > 0:
        apply_feature_extraction(theta_cvg)
        fig_src3, fig_src4 = apply_X(n_bin, w, n_clusters)
        return XPVI_figures(fig_src3, fig_src4)

    # @app.callback(
    #     Output("output-data-upload7", "children"),
    #     Input("XPVI_run", "n_clicks"),
    #     State("n_bins", "value"),
    #     State("w", "value"),
    #     State("n_clusters", "value")
    #     )
    # def plot_Xdata(n,n_bin, w, n_clusters):
    #     if n > 0:
    #         fig_src3, fig_src4 = apply_X(n_bin, w, n_clusters)
    #         return XPVI_figures(fig_src3, fig_src4)

    @app.callback(
        Output("output-data-upload110", "children"),
        Input("decl2NL_framework", "n_clicks"),
    )
    def X2NL(n):
        if n > 0:
            return decl2NL_parameters()

    @app.callback(
        Output("output-data-upload109", "children"),
        Input("decl2NL_pars", "n_clicks"),
        State("cluster_number", "value"),
        State("segment_number", "value")
    )
    def X2NL_calc(n, cluster, segment):
        if n > 0:
            list_sorted, list_sorted_reverse = decl2NL(cluster, segment)
            return statistics_print(list_sorted, list_sorted_reverse)

    @app.callback(
        Output("log-display", "children"),
        Input("latest_log", "n_clicks"),
    )
    def update_logs(n):
        if n > 0:
            # Read the log file and return its contents
            log_file = "log.log"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    logs = f.read()
            else:
                logs = "No logs yet."
            return logs
