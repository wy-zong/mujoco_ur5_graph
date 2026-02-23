python -m lerobot.scripts.lerobot_edit_dataset `
    --repo_id "full-fold-the-rag-parquet-merged" `
    --operation.type merge `
    --operation.repo_ids "['full-fold-the-rag-parquet','full-fold-the-rag-parquet-c']"

驗證合併是否正確完成
python C:\Users\ccu\mujoco_ur5_graph\scripts\validate_merged_dataset.py `
    --base-dir "C:\Users\ccu\mujoco_ur5_graph\outputs" `
    --src1 "full-fold-the-rag-parquet" `
    --src2 "full-fold-the-rag-parquet-c" `
    --merged "full-fold-the-rag-parquet-merged" `
    --sample-rows 20