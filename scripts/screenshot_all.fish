#!/usr/bin/env fish

set SCRIPT_DIR (dirname (realpath (status --current-filename)))
set REPO_DIR (dirname $SCRIPT_DIR)
set CULL_BIN $REPO_DIR/build/release/cull
set ASSETS_DIR $REPO_DIR/assets/Models
set SCREENSHOTS_DIR $REPO_DIR/screenshots

if not test -x $CULL_BIN
    echo "error: binary not found at $CULL_BIN — run: cd build/release && ninja"
    exit 1
end

mkdir -p $SCREENSHOTS_DIR

set batch_pids
set batch_names

for model_dir in $ASSETS_DIR/*/
    set model_name (basename $model_dir)
    set out_path $SCREENSHOTS_DIR/$model_name.png

    set gltf_path ""
    for subdir in glTF GLB glTF-Binary glTF-Embedded
        set candidate (
            find "$model_dir/$subdir" -maxdepth 1 \
                \( -name "*.gltf" -o -name "*.glb" \) 2>/dev/null | head -1
        )
        if test -n "$candidate"
            set gltf_path $candidate
            break
        end
    end

    if test -z "$gltf_path"
        echo "[$model_name] no gltf found, skipping"
        continue
    end

    echo "[$model_name] launching..."
    timeout 30 $CULL_BIN "$gltf_path" --screenshot "$out_path" &
    set batch_pids $batch_pids $last_pid
    set batch_names $batch_names $model_name

    if test (count $batch_pids) -ge 16
        for i in (seq (count $batch_pids))
            wait $batch_pids[$i]
            if test $status -ne 0
                echo "[$batch_names[$i]] FAILED"
            else
                echo "[$batch_names[$i]] done"
            end
        end
        set batch_pids
        set batch_names
    end
end

# flush remaining
for i in (seq (count $batch_pids))
    wait $batch_pids[$i]
    if test $status -ne 0
        echo "[$batch_names[$i]] FAILED"
    else
        echo "[$batch_names[$i]] done"
    end
end

echo "done — screenshots in $SCREENSHOTS_DIR"
