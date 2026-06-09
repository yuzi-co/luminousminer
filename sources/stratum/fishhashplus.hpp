#pragma once

#include <network/network.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    // KarlsenHashV2 (FishHashPlus) over the Kaspa gostratum dialect (karlsen-stratum-bridge,
    // forked from kaspa-stratum-bridge). Same wire protocol family as kHeavyHash; only the PoW
    // differs (handled by the resolver/kernel).
    //   mining.notify params = [ jobIdStr, [4 LE u64 pre_pow words], timestamp ]   (NORMAL)
    //                       or [ jobIdStr, "<80 hex>" ]                            (BIG)
    //   mining.submit params = [ wallet.worker, jobIdStr, nonceHex ].
    struct StratumFishhashPlus : public stratum::Stratum
    {
      public:
        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void onMiningSetExtraNonce(boost::json::object const& root) final;
        void onUnknownMethod(boost::json::object const& root) final;

        void miningSubscribe() override;
        // The FishHash resolver submits an object {jobId, nonce}; we translate it to the
        // Kaspa array wire form.
        void miningSubmit(uint32_t const deviceId, boost::json::object const& params) final;
    };
}
