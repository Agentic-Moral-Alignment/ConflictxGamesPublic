declare -a personal=("autonomy" "authenticity" "creativity" "empowerment")
declare -a protective=("responsibility" "harmlessness" "compliance" "privacy")

for p in "${personal[@]}"
    do
        for v in "${protective[@]}"
            do
                python -m conflictxgame.solo.generate_scenarios -o data/personalprotective/scenarios -m claude-3-5-sonnet-latest -v personalprotective -n 240 -d -dt 0.8 -b 40 --max-tokens 4096 -v1 $p -v2 $v
                python -m conflictxgame.solo.filter_scenarios -i data/personalprotective/scenarios/claude-3-5-sonnet-latest_${p}_${v}.json -o data/personalprotective/outputs -m gpt-4.1 -v personalprotective
            done
    done
