`timescale 1ns/1ps
// INT8 MLP inference core (784-32-10), one shared MAC, sequential.
// Image is written through the img_* port BEFORE pulsing start.
module nn_core #(
    parameter integer M1_INT = 6705,   // layer-1 requantization multiplier
    parameter integer SHIFT  = 24      // fixed-point shift
)(
    input  wire         clk,
    input  wire         rst,
    input  wire         start,
    input  wire         img_we,        // image write port
    input  wire [9:0]   img_addr,
    input  wire [7:0]   img_data,      // pixel, 0..127
    output reg          done,
    output reg  [3:0]   pred,
    output wire [319:0] scores_flat    // 10 x signed 32-bit layer-2 scores
);

    // ---------------- memories ----------------
    reg signed [7:0]  w1_mem [0:25087];   // 32 x 784, neuron-major
    reg signed [31:0] b1_mem [0:31];
    reg signed [7:0]  w2_mem [0:319];     // 10 x 32, neuron-major
    reg signed [31:0] b2_mem [0:9];
    reg        [7:0]  img_mem [0:783];
    reg        [7:0]  h_mem  [0:31];      // hidden activations (0..127)
    reg signed [31:0] score  [0:9];

    initial begin
        $readmemh("w1.mem", w1_mem);
        $readmemh("b1.mem", b1_mem);
        $readmemh("w2.mem", w2_mem);
        $readmemh("b2.mem", b2_mem);
    end

    // ---------------- control registers ----------------
    localparam [2:0] S_IDLE=3'd0, S_RUN=3'd1, S_FLUSH=3'd2,
                     S_FIN=3'd3, S_ARGMAX=3'd4, S_DONE=3'd5;

    reg [2:0]         state;
    reg               layer;       // 0 = hidden layer, 1 = output layer
    reg [5:0]         n;           // neuron index
    reg [9:0]         i;           // input index
    reg [14:0]        ptr;         // weight address (runs 0..N-1 sequentially)
    reg               vld;         // data from memories is valid for the MAC
    reg signed [31:0] acc;
    reg [3:0]         k;           // argmax counter
    reg signed [31:0] best;

    wire [9:0] last_i = layer ? 10'd31 : 10'd783;
    wire [5:0] last_n = layer ? 6'd9   : 6'd31;

    // ---------------- synchronous memory reads (1-cycle latency) ----------------
    reg signed [7:0] w1_q, w2_q;
    reg        [7:0] xi_q, xh_q;

    always @(posedge clk) begin
        if (img_we) img_mem[img_addr] <= img_data;
        xi_q <= img_mem[i];
        xh_q <= h_mem[i[4:0]];
        w1_q <= w1_mem[ptr];
        w2_q <= w2_mem[ptr[8:0]];
    end

    // ---------------- MAC ----------------
    wire        [7:0]  x_mac    = layer ? xh_q : xi_q;
    wire signed [7:0]  w_mac    = layer ? w2_q : w1_q;
    wire signed [16:0] mac_prod = $signed({1'b0, x_mac}) * w_mac;

    // ---------------- finish-neuron math (bias, requantize, clip) ----------------
    wire signed [31:0] bias    = layer ? b2_mem[n[3:0]] : b1_mem[n[4:0]];
    wire signed [31:0] sum     = acc + bias;
    wire signed [47:0] rq_prod = sum * M1_INT;                       // 48-bit product
    wire signed [47:0] rq_rnd  = rq_prod + (48'sd1 <<< (SHIFT-1));   // rounding
    wire signed [47:0] rq_sh   = rq_rnd >>> SHIFT;                   // arithmetic shift
    wire        [7:0]  h_clip  = (rq_sh < 0)   ? 8'd0   :
                                 (rq_sh > 127) ? 8'd127 : rq_sh[7:0];

    // ---------------- main FSM ----------------
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; done <= 1'b0; pred <= 4'd0;
            layer <= 1'b0; n <= 6'd0; i <= 10'd0; ptr <= 15'd0;
            acc <= 32'sd0; vld <= 1'b0; k <= 4'd0; best <= 32'sd0;
        end else begin
            if (vld) acc <= acc + mac_prod;

            case (state)
                S_IDLE, S_DONE: begin
                    if (start) begin
                        done <= 1'b0; layer <= 1'b0; n <= 6'd0;
                        i <= 10'd0; ptr <= 15'd0; acc <= 32'sd0; vld <= 1'b0;
                        state <= S_RUN;
                    end
                end

                S_RUN: begin                  // issue one read per cycle
                    vld <= 1'b1;
                    ptr <= ptr + 15'd1;
                    if (i == last_i) begin i <= 10'd0; state <= S_FLUSH; end
                    else             i <= i + 10'd1;
                end

                S_FLUSH: begin                // last product gets accumulated
                    vld   <= 1'b0;
                    state <= S_FIN;
                end

                S_FIN: begin                  // neuron sum is complete
                    acc <= 32'sd0;
                    if (layer == 1'b0) h_mem[n[4:0]]  <= h_clip;
                    else               score[n[3:0]]  <= sum;
                    if (n == last_n) begin
                        n <= 6'd0;
                        if (layer == 1'b0) begin
                            layer <= 1'b1; ptr <= 15'd0; state <= S_RUN;
                        end else begin
                            k <= 4'd0; state <= S_ARGMAX;
                        end
                    end else begin
                        n <= n + 6'd1; state <= S_RUN;
                    end
                end

                S_ARGMAX: begin               // first maximum wins (same as numpy)
                    if (k == 4'd0 || score[k] > best) begin
                        best <= score[k]; pred <= k;
                    end
                    if (k == 4'd9) begin state <= S_DONE; done <= 1'b1; end
                    else k <= k + 4'd1;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    genvar g;
    generate
        for (g = 0; g < 10; g = g + 1) begin : sf
            assign scores_flat[g*32 +: 32] = score[g];
        end
    endgenerate
endmodule
