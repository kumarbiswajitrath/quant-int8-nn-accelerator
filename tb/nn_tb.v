`timescale 1ns/1ps
module nn_tb;
    localparam NUM = 200;   // images available in test_images.mem
    localparam RUN = 200;   // how many to simulate (use 5 for a quick first run)

    reg clk = 0, rst = 1, start = 0, img_we = 0;
    reg [9:0] img_addr = 0;
    reg [7:0] img_data = 0;
    wire done;
    wire [3:0] pred;
    wire [319:0] scores_flat;

    nn_core dut (.clk(clk), .rst(rst), .start(start), .img_we(img_we),
                 .img_addr(img_addr), .img_data(img_data),
                 .done(done), .pred(pred), .scores_flat(scores_flat));

    always #5 clk = ~clk;   // 100 MHz

    reg [7:0] imgs   [0:NUM*784-1];
    reg [3:0] labels [0:NUM-1];
    reg signed [31:0] gold [0:9];
    integer g0,g1,g2,g3,g4,g5,g6,g7,g8,g9;
    integer fd, r, img, j, k, cyc, exact, correct, bad_scores, first_cyc;
    reg signed [31:0] s;

    initial begin
        $readmemh("test_images.mem", imgs);
        $readmemh("test_labels.txt", labels);   // single digits: same in hex
        fd = $fopen("golden_scores.txt", "r");
        exact = 0; correct = 0; first_cyc = 0;

        repeat (4) @(negedge clk);
        rst = 0;
        repeat (2) @(negedge clk);

        for (img = 0; img < RUN; img = img + 1) begin
            // 1) load image
            for (j = 0; j < 784; j = j + 1) begin
                @(negedge clk);
                img_we = 1; img_addr = j; img_data = imgs[img*784 + j];
            end
            @(negedge clk); img_we = 0;

            // 2) start and wait for done
            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            cyc = 0;
            while (!done && cyc < 200000) begin @(negedge clk); cyc = cyc + 1; end
            if (img == 0) first_cyc = cyc;

            // 3) compare with golden model
            r = $fscanf(fd, "%d %d %d %d %d %d %d %d %d %d\n",
                        g0,g1,g2,g3,g4,g5,g6,g7,g8,g9);
            gold[0]=g0; gold[1]=g1; gold[2]=g2; gold[3]=g3; gold[4]=g4;
            gold[5]=g5; gold[6]=g6; gold[7]=g7; gold[8]=g8; gold[9]=g9;
            bad_scores = 0;
            for (k = 0; k < 10; k = k + 1) begin
                s = scores_flat[k*32 +: 32];
                if (s !== gold[k]) bad_scores = bad_scores + 1;
            end
            if (bad_scores == 0) exact = exact + 1;
            else $display("MISMATCH image %0d: %0d of 10 scores differ", img, bad_scores);
            if (pred == labels[img]) correct = correct + 1;
        end

        $display("--------------------------------------------");
        $display("Images simulated          : %0d", RUN);
        $display("Bit-exact score matches   : %0d / %0d", exact, RUN);
        $display("Correct class vs label    : %0d / %0d", correct, RUN);
        $display("Cycles per inference      : %0d", first_cyc);
        $display("--------------------------------------------");
        $finish;
    end
endmodule
