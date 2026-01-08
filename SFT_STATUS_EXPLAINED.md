# Fine-Tuning Job Status Explained

## Job States

When you see "Running" status in the Fireworks dashboard, here's what it means:

### **JOB_STATE_RUNNING / Running**
✅ **Good!** The fine-tuning job is actively training the model.
- The model is learning from your dataset
- Training loss is being optimized
- This typically takes **10-60 minutes** depending on:
  - Dataset size (yours: 250 examples - should be relatively quick)
  - Model size (Qwen 7B is medium-sized)
  - Number of epochs (you're using 1 epoch)
  - LoRA rank (you're using rank 16)

**What to expect:**
- Job will progress through training steps
- You can monitor progress in the dashboard
- Loss metrics should decrease over time
- When complete, status changes to "COMPLETED"

### Other Possible States:

**JOB_STATE_CREATING**
- Job is being set up
- Resources are being allocated
- Usually very brief (seconds to minutes)

**JOB_STATE_COMPLETED**
- ✅ Training finished successfully!
- Model is ready to use
- You can now deploy it for inference

**JOB_STATE_FAILED**
- ❌ Something went wrong
- Check error logs in the dashboard
- Common causes: dataset format issues, model compatibility, resource limits

**JOB_STATE_CANCELLED**
- Job was manually cancelled
- Can be restarted if needed

**JOB_STATE_VALIDATING**
- Dataset is being validated
- Checking format and content
- Usually happens before training starts

## Monitoring Your Job

**In Dashboard:**
- Visit: `https://app.fireworks.ai/dashboard/fine-tuning/supervised/{job_id}`
- See real-time progress
- View training metrics (loss, steps)
- Check estimated completion time

**Via Script:**
- The script monitors automatically if you let it run
- Or check manually: `python3 run_sft_workflow.py --skip-baseline --skip-upload --skip-tuning --model-id your-model-id`

## Expected Timeline

For your setup (250 examples, Qwen 7B, 1 epoch):
- **Dataset Upload:** ✅ Already done (~1 minute)
- **Job Creation:** ✅ Already done (~30 seconds)
- **Training (Running):** ⏳ **10-30 minutes** (currently here)
- **Completion:** Will change to "COMPLETED" when done
- **Deployment:** Manual or automatic (depends on settings)
- **Evaluation:** 5-15 minutes

**Total:** ~30-60 minutes from start to finish

## What Happens Next

Once status changes to "COMPLETED":
1. Model will be available at: `accounts/{account_id}/models/{your-model-id}`
2. You can deploy it for inference
3. Run post-tuning evaluation to compare results
4. Check if accuracy/latency improved!

## Tips

- **Don't cancel** the job while it's running - let it complete
- **Check dashboard** for detailed progress and metrics
- **Be patient** - training takes time but is worth it!
- **Monitor logs** if you see any issues

