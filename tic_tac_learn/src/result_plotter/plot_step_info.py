import pandas as pd 
from recordclass import RecordClass
import logging


def plot_step_info(rc: RecordClass,
                       save_path :str, 
                       save:bool = True) -> list:
    
    all_res = pd.DataFrame(columns=["games_played",
                                    "step_lr",
                                    "step_test_wins",
                                    "step_test_draws",
                                    "step_test_losses"])
    for games, info  in rc.overall_res.items():
        #learning rate
        lr_c =  info[0]
        #total wins this step
        tw_c = info[1]
        #total draws this step
        td_c = info[2]
        #test games this step
        tg_c = info[3]
        #total games lost this step
        tl_c  = tg_c -td_c -tw_c
        wrk_df = pd.DataFrame([[games,lr_c,tw_c,td_c,tg_c,tl_c]], columns=["games_played",
                                    "step_lr",
                                    "step_test_wins",
                                    "step_test_draws",
                                    "step_test_games",
                                    "step_test_losses"])
        
        all_res =  pd.concat([all_res,wrk_df])

    logging.info(f"plot step info - all_res head  {all_res.head()}")
    
    logging.info(f"plot step info - all_res columns  {all_res.columns}")

    rc = all_res.plot(kind="scatter",x="games_played" , y= "step_lr").get_figure()
    rc.savefig(save_path+"learning_rate_change.png")

    gw = all_res.plot(kind="scatter",x="games_played" , y= "step_test_wins").get_figure()
    gw.savefig(save_path+"test_games_won.png")

    gd = all_res.plot(kind="scatter",x="games_played" , y= "step_test_draws").get_figure()
    gd.savefig(save_path+"test_games_drawn.png")

    gl = all_res.plot(kind="scatter",x="games_played" , y= "step_test_losses").get_figure()
    gl.savefig(save_path+"test_games_losses.png")

    return {"Learning Rate Change":rc,
            "Test games Won":  gw,
            "Test Games Drawn": gd,
            "Test Games Lost": gl}